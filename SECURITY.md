# Security policy

## Reporting a vulnerability

Please **do not open a public issue**. Use GitHub's private vulnerability reporting
("Security" tab → "Report a vulnerability") on this repository. Include steps to reproduce,
the affected version and your platform. You should get an answer within a week.

## Supported versions

Only the latest released minor version receives security fixes while the project is in
0.x.

## Notes

* Model weights are downloaded by the upstream libraries (GigaAM, `qwen-tts` via Hugging
  Face, Silero via the `silero` package or, as a fallback, `torch.hub` from
  `snakers4/silero-models`, which executes that repository's loader code). Pin versions
  and pre-download models if your deployment needs to stay offline.
* Voice cloning reproduces a real person's voice. Only clone voices you have permission
  to use.
* The package sends no telemetry and makes no network calls of its own.
