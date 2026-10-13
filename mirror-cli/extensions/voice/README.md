# Local voice dictation

Enable **voice** in `/extensions`. Hold Space to record; release to transcribe
into the draft without submitting. **Ctrl+B** or `/voice` toggles recording;
Ctrl+B also cancels setup or transcription. Capture stops after two minutes.
Key repeat must be enabled; isolated Space taps can take 1.5 seconds to appear.

## Setup

- Requires `uv >=0.10.10` on PATH. macOS needs terminal microphone permission;
  Linux needs PortAudio. Native Windows is unsupported.
- First use each session asks permission (default: **Not now**). Initial setup
  installs locked wheels and downloads the pinned Whisper base model (~150 MB).
  It never downloads uv or Python, builds packages, or opens the microphone.
- Ready environments run offline. Audio stays on the CLI host; SSH does not
  forward the microphone. Each utterance starts Python and loads the CPU model.

## State and trust

Environments live in `~/.mirror/extension-environments/<fingerprint>/`; dependency, worker or
interpreter changes require setup again. To repair a broken cache, stop Mirror
and remove the affected directory. Old environments are not cleaned automatically.

Helpers receive no inherited authentication variables, but **are not sandboxed**:
they retain the user's filesystem access. Trust the dependencies, not just their
hashes. Dependency/model updates require reviewing `uv.lock`, the pinned revision
and SHA-256 hashes in `worker.py`, then testing setup and offline reuse.
