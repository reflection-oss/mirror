"""Record one utterance and transcribe locally; launched in a separate uv environment."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import threading
from typing import cast

import faster_whisper
import numpy as np
import sounddevice
from faster_whisper import utils

# Pin both the repository revision and bytes before parsing model artifacts.
_MODEL_REVISION = 'ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66'
_MODEL_FILES = {
  'config.json': '56a6d8110d311f19c8f0471e562832c7527f146b567275bfca59fcf7c184da9a',
  'model.bin': 'd01c3014881c9c6f3133c182f3d2887eb6ca1c789a7538c5c007196857a0a6a9',
  'tokenizer.json': 'fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab',
  'vocabulary.txt': '34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913',
}
_SAMPLE_RATE = 16000
_BLOCK_SIZE = 1600
_MAX_BLOCKS = 120 * _SAMPLE_RATE // _BLOCK_SIZE


def _stop_requested(stop: threading.Event) -> None:
  sys.stdin.readline()
  stop.set()


def record() -> bytes:
  """Capture mono PCM until stdin receives a line, closes, or two minutes pass."""
  stop = threading.Event()
  threading.Thread(target=_stop_requested, args=(stop,), daemon=True).start()
  chunks: list[bytes] = []
  sample_count = 0
  with sounddevice.RawInputStream(
    samplerate=_SAMPLE_RATE, channels=1, dtype='int16', blocksize=_BLOCK_SIZE
  ) as stream:
    print(json.dumps({'status': 'recording'}), flush=True)
    for _ in range(_MAX_BLOCKS):
      if stop.is_set():
        break
      block, overflowed = stream.read(_BLOCK_SIZE)  # pyright: ignore[reportUnknownMemberType]  # sounddevice's frames parameter is untyped.
      if overflowed:
        raise RuntimeError('Microphone overflow; try recording again.')
      chunk = bytes(block)
      chunks.append(chunk)
      samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768
      sample_count += samples.size
      print(
        json.dumps({
          'level': float(np.sqrt(np.mean(samples * samples))),
          'elapsed': sample_count / _SAMPLE_RATE,
        }),
        flush=True,
      )
  return b''.join(chunks)


def _model(*, local_files_only: bool) -> faster_whisper.WhisperModel:
  """Verify pinned bytes before any model parser consumes them."""
  directory = pathlib.Path(
    cast(
      str,
      utils.download_model(
        'Systran/faster-whisper-base',
        revision=_MODEL_REVISION,
        local_files_only=local_files_only,
        use_auth_token=False,
      ),
    )
  )
  for name, expected in _MODEL_FILES.items():
    with (directory / name).open('rb') as artifact:
      actual = hashlib.file_digest(artifact, 'sha256').hexdigest()
    if actual != expected:
      raise ValueError(f'Voice model integrity check failed: {name}')
  return faster_whisper.WhisperModel(
    str(directory), device='cpu', compute_type='int8', local_files_only=True
  )


def main() -> None:
  """Keep stdout exclusively for newline-delimited status and transcript JSON."""
  preparing = '--prepare' in sys.argv[1:]
  model = _model(local_files_only=not preparing or '--offline' in sys.argv[1:])
  if preparing:
    return
  recording = record()
  print(json.dumps({'status': 'transcribing'}), flush=True)
  parts: list[str] = []
  if recording:
    audio = np.frombuffer(recording, dtype=np.int16).astype(np.float32) / 32768
    duration = audio.size / _SAMPLE_RATE
    segments, _ = model.transcribe(audio, vad_filter=True)  # pyright: ignore[reportUnknownMemberType]  # Upstream vad_parameters has an unparameterized dict.
    progress = 0
    for segment in segments:
      parts.append(segment.text)
      # VAD-restored timestamps measure coverage of the original recording.
      percent = min(99, int(segment.end / duration * 100))
      if percent > progress:
        progress = percent
        print(json.dumps({'progress': progress}), flush=True)
  print(json.dumps({'progress': 100}), flush=True)
  print(json.dumps({'text': ''.join(parts).strip()}), flush=True)


if __name__ == '__main__':
  main()
