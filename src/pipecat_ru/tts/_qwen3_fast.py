"""Fast streaming generation for Qwen3-TTS 12 Hz models: static KV cache + CUDA graphs.

``qwen-tts`` generates with Hugging Face ``generate()``. Every 80 ms codec frame takes
one talker step and 15 sub-talker (code predictor) steps, each launching hundreds of
small kernels from Python. On a fast GPU that loop is launch-bound: an RTX 5090 sits
at about 8% utilization and runs slower than real time.

This module keeps ``qwen-tts`` for the weights, the tokenizer and prompt building, and
replaces the generation loop:

* the talker and the code predictor run on fixed-size KV caches;
* one codec frame (15 code-predictor steps, one talker step and all sampling) is
  captured as a single CUDA graph and replayed, so a frame costs one launch;
* frames are decoded to audio as they are produced, with left context for the causal
  codec decoder (``qwen-tts`` itself decodes in chunks with 25 frames of context).

Sampling follows ``qwen-tts``: temperature, top-k, top-p, repetition penalty on the
first codebook, suppressed control tokens and at least two frames before EOS. Random
draws differ from Hugging Face's, so outputs are not bit-identical; with
``do_sample=False`` the codes match up to floating-point noise.

Only the torch and ``qwen-tts`` objects already loaded by :class:`Qwen3TTSService` are
used; nothing here is imported until the fast path is switched on.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from pipecat_ru._audio import FloatArray

#: Frames of left context fed to the codec decoder for each streamed chunk.
DECODE_CONTEXT_FRAMES = 25


class _PromptCaptured(Exception):
    """Raised from the patched ``talker.generate`` once the prompt is built."""


@dataclass(frozen=True)
class _Sampling:
    do_sample: bool
    top_k: int
    top_p: float
    temperature: float


@dataclass(frozen=True)
class _GraphKey:
    talker: _Sampling
    sub: _Sampling
    repetition_penalty: float


class _StaticCache:
    """KV cache with fixed buffers, written with ``index_copy_`` (graph-safe).

    Implements the one method the ``qwen-tts`` attention layers call.
    """

    def __init__(
        self,
        torch: Any,
        layers: int,
        kv_heads: int,
        head_dim: int,
        max_len: int,
        dtype: Any,
        device: Any,
    ) -> None:
        shape = (1, kv_heads, max_len, head_dim)
        self.keys = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(layers)]
        self.values = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(layers)]

    def update(
        self,
        key_states: Any,
        value_states: Any,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        assert cache_kwargs is not None
        pos = cache_kwargs["cache_position"]
        self.keys[layer_idx].index_copy_(2, pos, key_states)
        self.values[layer_idx].index_copy_(2, pos, value_states)
        return self.keys[layer_idx], self.values[layer_idx]


def _sample(torch: Any, logits: Any, s: _Sampling) -> Any:
    """Sample one token per row from float logits ``(1, V)``; graph-capturable."""
    if not s.do_sample:
        return logits.argmax(-1)
    logits = logits / s.temperature
    if s.top_k > 0:
        kth = torch.topk(logits, min(s.top_k, logits.shape[-1]), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if s.top_p < 1.0:
        sorted_logits, idx = torch.sort(logits, descending=False, dim=-1)
        remove = sorted_logits.softmax(-1).cumsum(-1) <= (1.0 - s.top_p)
        remove[..., -1:] = False
        logits = logits.masked_fill(remove.scatter(-1, idx, remove), float("-inf"))
    probs = logits.softmax(-1)
    # Exponential race == sampling from probs, without multinomial's host sync.
    noise = torch.empty_like(probs).exponential_()
    return (probs / noise).argmax(-1)


class Qwen3FastGenerator:
    """Streams audio from a loaded ``qwen_tts.Qwen3TTSModel`` using CUDA graphs.

    Args:
        tts: The loaded ``Qwen3TTSModel`` wrapper (12 Hz model, on a CUDA device).
        max_seq_len: Talker KV-cache length: prompt plus generated frames
            (12.5 frames per second of audio).
        cuda_graphs: Capture frames as CUDA graphs. ``False`` runs the same static-cache
            code eagerly (slower; for debugging).
        decode_context: Frames of left context for each decoded chunk.
    """

    def __init__(
        self,
        tts: Any,
        *,
        max_seq_len: int = 2048,
        cuda_graphs: bool = True,
        decode_context: int = DECODE_CONTEXT_FRAMES,
    ) -> None:
        import torch

        self._torch = torch
        self._tts = tts
        model = tts.model
        self._model = model
        talker = model.talker
        self._talker = talker
        self._tm = talker.model
        self._cp = talker.code_predictor
        self._cpm = self._cp.model
        decoder = getattr(getattr(model.speech_tokenizer, "model", None), "decoder", None)
        if decoder is None or not hasattr(decoder, "total_upsample"):
            raise ValueError("Qwen3FastGenerator supports the 12 Hz Qwen3-TTS models only")
        self._decoder = decoder
        self.frame_samples = int(decoder.total_upsample)
        self.sample_rate = int(model.speech_tokenizer.get_output_sample_rate())

        param = next(talker.parameters())
        self._device = param.device
        self._dtype = param.dtype
        if self._device.type != "cuda":
            cuda_graphs = False
        self._cuda_graphs = cuda_graphs
        self._decode_context = decode_context
        self._max_len = max_seq_len
        self._lock = threading.Lock()

        tcfg = talker.config
        ccfg = self._cp.config
        self._groups = int(tcfg.num_code_groups)
        self._eos = int(tcfg.codec_eos_token_id)
        vocab = int(tcfg.vocab_size)
        dev, dt = self._device, self._dtype

        with torch.inference_mode():
            self._t_cache = _StaticCache(
                torch,
                tcfg.num_hidden_layers,
                tcfg.num_key_value_heads,
                tcfg.head_dim,
                max_seq_len,
                dt,
                dev,
            )
            cp_len = self._groups  # positions 0..15: talker hidden + 15 codes
            self._cp_cache = _StaticCache(
                torch,
                ccfg.num_hidden_layers,
                ccfg.num_key_value_heads,
                ccfg.head_dim,
                cp_len,
                dt,
                dev,
            )
            dummy = torch.zeros(1, dtype=dt, device=dev)
            t_pos = torch.arange(max_seq_len, device=dev).view(1, 1, -1).expand(3, 1, -1)
            self._t_cos, self._t_sin = self._tm.rotary_emb(dummy, t_pos)
            self._t_arange = torch.arange(max_seq_len, device=dev)
            cp_pos = torch.arange(cp_len, device=dev)
            self._cp_pos = cp_pos
            self._cp_cos, self._cp_sin = self._cpm.rotary_emb(dummy, cp_pos.view(1, -1))
            causal = cp_pos.view(1, -1) <= cp_pos.view(-1, 1)  # (q, k)
            self._cp_masks = {0: causal[0:2].view(1, 1, 2, cp_len)}
            for start in range(2, cp_len):
                self._cp_masks[start] = causal[start : start + 1].view(1, 1, 1, cp_len)

            # Static inputs/outputs of the frame graph.
            hidden = int(tcfg.hidden_size)
            self._h = torch.zeros(1, 1, hidden, dtype=dt, device=dev)
            self._text_add = torch.zeros(1, 1, hidden, dtype=dt, device=dev)
            self._code0 = torch.zeros(1, dtype=torch.long, device=dev)
            self._pos = torch.zeros(1, dtype=torch.long, device=dev)
            self._codes = torch.zeros(self._groups, dtype=torch.long, device=dev)
            self._seen = torch.zeros(vocab, dtype=torch.bool, device=dev)
            self._eos_bias = torch.zeros(vocab, dtype=torch.float32, device=dev)
            self._suppress = torch.zeros(vocab, dtype=torch.float32, device=dev)
            self._suppress_ids: tuple[int, ...] | None = None

        self._key: _GraphKey | None = None
        self._graphs: dict[_GraphKey, Any] = {}

    # ------------------------------------------------------------------ forward passes

    def _run_talker(self, x: Any, pos: Any, mask: Any) -> tuple[Any, Any]:
        cos = self._t_cos.index_select(2, pos)
        sin = self._t_sin.index_select(2, pos)
        for layer in self._tm.layers:
            x = layer(
                x,
                attention_mask=mask,
                past_key_values=self._t_cache,
                cache_position=pos,
                position_embeddings=(cos, sin),
            )[0]
        x = self._tm.norm(x)
        return x, self._talker.codec_head(x[:, -1]).float()

    def _run_cp(self, x: Any, start: int, length: int) -> Any:
        pos = self._cp_pos[start : start + length]
        cos = self._cp_cos[:, start : start + length]
        sin = self._cp_sin[:, start : start + length]
        mask = self._cp_masks[start]
        for layer in self._cpm.layers:
            x = layer(
                x,
                attention_mask=mask,
                past_key_values=self._cp_cache,
                cache_position=pos,
                position_embeddings=(cos, sin),
            )[0]
        return self._cpm.norm(x)

    def _sample_talker(self, logits: Any, key: _GraphKey) -> Any:
        torch = self._torch
        p = key.repetition_penalty
        if p != 1.0:
            penalized = torch.where(logits < 0, logits * p, logits / p)
            logits = torch.where(self._seen, penalized, logits)
        logits = logits + self._suppress + self._eos_bias
        tok = _sample(torch, logits, key.talker)
        self._seen.index_fill_(0, tok, True)
        return tok

    def _frame_step(self) -> None:
        """One codec frame: 15 code-predictor steps, then the next talker step."""
        torch = self._torch
        key = self._key
        assert key is not None
        cp, cpm = self._cp, self._cpm
        e0 = self._tm.codec_embedding(self._code0.view(1, 1))  # (1, 1, H)
        x = cp.small_to_mtp_projection(torch.cat([self._h, e0], dim=1))
        out = self._run_cp(x, 0, 2)
        tok = _sample(torch, cp.lm_head[0](out[:, -1]).float(), key.sub)
        toks = [self._code0, tok]
        acc = e0[:, 0] + cpm.codec_embedding[0](tok)
        for i in range(1, self._groups - 1):
            x = cp.small_to_mtp_projection(cpm.codec_embedding[i - 1](tok)).view(1, 1, -1)
            out = self._run_cp(x, i + 1, 1)
            tok = _sample(torch, cp.lm_head[i](out[:, -1]).float(), key.sub)
            toks.append(tok)
            acc = acc + cpm.codec_embedding[i](tok)
        self._codes.copy_(torch.cat(toks))

        mask = (self._t_arange <= self._pos).view(1, 1, 1, -1)
        hid, logits = self._run_talker(acc.view(1, 1, -1) + self._text_add, self._pos, mask)
        self._h.copy_(hid[:, -1:])
        self._code0.copy_(self._sample_talker(logits, key))
        self._pos.add_(1)

    # ------------------------------------------------------------------ graph management

    def _set_key(self, key: _GraphKey, suppress_ids: tuple[int, ...]) -> Any:
        """Select sampling settings; capture (or reuse) the frame graph for them."""
        if suppress_ids != self._suppress_ids:
            self._suppress.zero_()
            if suppress_ids:
                idx = self._torch.tensor(suppress_ids, device=self._device)
                self._suppress.index_fill_(0, idx, float("-inf"))
            self._suppress_ids = suppress_ids
        self._key = key
        if not self._cuda_graphs:
            return None
        graph = self._graphs.get(key)
        if graph is None:
            graph = self._capture()
            self._graphs[key] = graph
        return graph

    def _capture(self) -> Any:
        torch = self._torch
        self._pos.zero_()
        self._code0.zero_()
        self._h.zero_()
        stream = torch.cuda.Stream(device=self._device)
        stream.wait_stream(torch.cuda.current_stream(self._device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._frame_step()
        torch.cuda.current_stream(self._device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._frame_step()
        torch.cuda.synchronize(self._device)
        return graph

    def warmup(
        self,
        call: Callable[[], Any],
        *,
        ref_codes: Any = None,
        first_chunk_frames: int = 3,
        chunk_frames: int = 8,
    ) -> None:
        """Prepare for the first reply: capture the frame graph for ``call``'s sampling
        settings and run the decoder on first-chunk, full-chunk and tail shapes, so the
        first real reply does not pay for CUDA's lazy kernel loading (about 1.3 s).
        """
        frames = first_chunk_frames + chunk_frames + max(1, chunk_frames // 2)
        for _ in range(2):  # the second pass also covers a prompt length seen before
            for _ in self.stream(
                call,
                ref_codes=ref_codes,
                first_chunk_frames=first_chunk_frames,
                chunk_frames=chunk_frames,
                max_frames=frames,
            ):
                pass

    # ------------------------------------------------------------------ public API

    def _build_prompt(self, call: Callable[[], Any]) -> dict[str, Any]:
        """Run ``call`` (a ``qwen-tts`` generate_* call) until the talker prompt is ready."""
        talker = self._talker
        captured: dict[str, Any] = {}

        def grab(**kwargs: Any) -> Any:
            captured.update(kwargs)
            raise _PromptCaptured

        talker.generate = grab
        try:
            call()
        except _PromptCaptured:
            pass
        finally:
            del talker.generate
        if not captured:
            raise RuntimeError("qwen-tts did not reach talker.generate()")
        return captured

    def _decode(self, frames: list[Any], start: int, ref_codes: Any) -> FloatArray:
        torch = self._torch
        codes = torch.stack(frames)  # (N, Q)
        offset = 0
        if ref_codes is not None:
            codes = torch.cat([ref_codes, codes])
            offset = int(ref_codes.shape[0])
        first = offset + start
        ctx_start = max(0, first - self._decode_context)
        window = codes[ctx_start:].clamp(min=0).transpose(0, 1).unsqueeze(0)  # (1, Q, W)
        wav = self._decoder(window)[0, 0, (first - ctx_start) * self.frame_samples :]
        return np.ascontiguousarray(wav.float().cpu().numpy(), dtype=np.float32)

    def stream(
        self,
        call: Callable[[], Any],
        *,
        ref_codes: Any = None,
        first_chunk_frames: int = 3,
        chunk_frames: int = 8,
        max_frames: int | None = None,
    ) -> Iterator[FloatArray]:
        """Generate speech for one ``qwen-tts`` call, yielding float32 chunks.

        Args:
            call: A zero-argument function that calls ``generate_custom_voice`` or
                ``generate_voice_clone`` on the wrapper; it is stopped as soon as the
                prompt is built, so only the prompt-building part runs.
            ref_codes: Reference codec frames ``(T, 16)`` of an ICL voice-clone prompt;
                used as decoder left context for the first chunk.
            first_chunk_frames: Frames in the first audio chunk (80 ms each).
            chunk_frames: Frames in each later chunk.
            max_frames: Stop after this many frames (default: ``max_new_tokens``).
        """
        torch = self._torch
        with self._lock, torch.inference_mode():
            kw = self._build_prompt(call)
            embeds = kw["inputs_embeds"]
            mask = kw.get("attention_mask")
            if embeds.shape[0] != 1 or (mask is not None and not bool(mask.all())):
                raise ValueError("Qwen3FastGenerator generates one utterance at a time")
            key = _GraphKey(
                talker=_Sampling(
                    bool(kw["do_sample"]),
                    int(kw["top_k"] or 0),
                    float(kw["top_p"]),
                    float(kw["temperature"]),
                ),
                sub=_Sampling(
                    bool(kw["subtalker_dosample"]),
                    int(kw["subtalker_top_k"] or 0),
                    float(kw["subtalker_top_p"]),
                    float(kw["subtalker_temperature"]),
                ),
                repetition_penalty=float(kw.get("repetition_penalty") or 1.0),
            )
            eos = int(kw.get("eos_token_id", self._eos))
            limit = int(kw.get("max_new_tokens", 4096))
            if max_frames is not None:
                limit = min(limit, max_frames)
            graph = self._set_key(key, tuple(kw.get("suppress_tokens") or ()))
            trailing = kw["trailing_text_hidden"].to(self._dtype)
            pad = kw["tts_pad_embed"].to(self._dtype)
            if ref_codes is not None:
                ref_codes = ref_codes.to(self._device).long()
                if ref_codes.dim() != 2:
                    ref_codes = None

            # Prefill (eager): prompt into the talker cache, first-codebook token.
            length = int(embeds.shape[1])
            if length + limit + 1 > self._max_len:
                limit = self._max_len - length - 1
                if limit <= 0:
                    raise ValueError(f"prompt of {length} tokens exceeds max_seq_len")
            pos = self._t_arange[:length]
            pmask = (self._t_arange.view(1, -1) <= pos.view(-1, 1)).view(1, 1, length, -1)
            self._seen.zero_()
            self._eos_bias.zero_()
            self._eos_bias[eos] = float("-inf")  # min_new_tokens=2
            hid, logits = self._run_talker(embeds.to(self._dtype), pos, pmask)
            self._h.copy_(hid[:, -1:])
            self._code0.copy_(self._sample_talker(logits, key))
            self._pos.fill_(length)

            frames: list[Any] = []
            emitted = 0
            n_text = int(trailing.shape[1])
            for step in range(limit):
                if int(self._code0.item()) == eos:
                    break
                self._text_add.copy_(trailing[:, step : step + 1] if step < n_text else pad)
                if graph is not None:
                    graph.replay()
                else:
                    self._frame_step()
                if step == 0:
                    self._eos_bias.zero_()
                frames.append(self._codes.clone())
                need = first_chunk_frames if emitted == 0 else chunk_frames
                if len(frames) - emitted >= need:
                    yield self._decode(frames, emitted, ref_codes)
                    emitted = len(frames)
            if len(frames) > emitted:
                yield self._decode(frames, emitted, ref_codes)
