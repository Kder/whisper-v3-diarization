# AMD Ryzen AI NPU Backend (whisper.cpp + VitisAI)

This fork adds a **pluggable transcription backend** that runs Whisper on the
**AMD Ryzen AI NPU (XDNA / XDNA2)** via whisper.cpp's VitisAI integration,
while keeping speaker diarization on CPU/GPU. Tested on a Ryzen AI Max+ 395
(Strix Halo, 50 TOPS XDNA2 NPU) on Windows 11.

To our knowledge this is the first open-source integration of
**"NPU transcription + speaker diarization"** in a single pipeline on AMD
Ryzen AI hardware.

## Why

- The original pipeline is CUDA-centric; on AMD machines transcription falls
  back to slow CPU inference.
- The XDNA2 NPU can run the Whisper encoder at low power, leaving the CPU
  free for alignment, diarization and clustering.
- whisper.cpp (built with `-DWHISPER_VITISAI=1`) offloads the encoder to the
  NPU and keeps full segment/word-level timestamps — exactly what the
  diarization stage needs.

## Architecture

```
audio ──► TranscriptionBackend (pluggable)
            ├── pytorch     → original HuggingFace Whisper (CUDA/CPU)
            └── whispercpp  → whisper.cpp, encoder on NPU via VitisAI
                                   │
                                   ▼ segments + timestamps
                     WhisperXDiarizer.diarize_with_transcript()
                        align (wav2vec2, CPU) → pyannote (CPU) → assign
```

When `--backend whispercpp` is selected, `diarize` **reuses the NPU
transcript** instead of letting WhisperX transcribe a second time on CPU.

## Prerequisites

1. **whisper.cpp built with VitisAI** (upstream source, do not use
   third-party repacks — they have shipped builds that silently fall back
   to CPU):

   ```bat
   cmake -B build -DWHISPER_VITISAI=1
   cmake --build build -j --config Release
   ```

2. **FlexML runtime (flexmlrt)** installed; the backend automatically
   injects `flexmlrt\lib` into `PATH` for the subprocess.

3. **Model + `.rai` encoder cache**: e.g. `ggml-large-v3-turbo.bin` with
   `ggml-large-v3-turbo-encoder-vitisai.rai` next to it (download via
   `models/download-vitisai-model.cmd`).

4. **pyannote model access** (for diarization): accept the terms at
   https://huggingface.co/pyannote/speaker-diarization-community-1 and set
   `HF_TOKEN`.

## Usage

```bash
# Transcribe on the NPU
audio-transcription transcribe meeting.mp3 --backend whispercpp --language zh

# Full pipeline: NPU transcription + CPU diarization
audio-transcription diarize meeting.mp3 --backend whispercpp --language zh \
    --min-speakers 2 --max-speakers 6
```

### Configuration

Resolved in order: constructor argument → environment variable → default.

| Setting | Env var | Default |
| --- | --- | --- |
| whisper-cli path | `WHISPER_CPP_EXE` | `D:\Programs\whisper.cpp\build\bin\Release\whisper-cli.exe` |
| ggml model path | `WHISPER_CPP_MODEL` | `D:\Programs\whisper.cpp\models\ggml-large-v3-turbo.bin` |
| flexmlrt root | `FLEXMLRT_ROOT` | `D:\Apps\flexmlrt` |

The backend verifies NPU offload actually happened (it looks for
`Vitis AI encoder model loaded` in the whisper.cpp log) and warns loudly if
the run silently fell back to CPU.

## Measured performance (Ryzen AI Max+ 395, Windows 11)

5-minute real-world Chinese meeting recording, end-to-end `diarize`:

| Stage | Runs on | Time | vs realtime |
| --- | --- | --- | --- |
| Transcription (large-v3-turbo) | NPU encoder + CPU decoder | ~98 s | ~3× |
| Word alignment (wav2vec2 zh) | CPU | ~32 s | ~9× |
| Diarization (pyannote community-1) | CPU | ~76 s | ~4× |
| **Total (300 s audio)** | | **~220 s** | **~1.4×** |

Notes:

- Whisper **large-v3 (full)** does not fit the NPU — use `large-v3-turbo`.
- The CPU decoder is the bottleneck of the NPU path on long audio.
- NPU software stack is currently production-grade on **Windows 11** only.

## Fixes included in this fork

- **whisperX ≥ 3.8 / pyannote-audio 4.x compatibility**:
  `DiarizationPipeline(token=...)` (was `use_auth_token=`), default model
  `pyannote/speaker-diarization-community-1`; version-adaptive fallback to
  the 3.1 API.
- **Speaker-label smoothing no longer collapses alternating speakers**:
  `smooth_speaker_labels` now only touches short segments, votes on original
  labels, and never flips on a tie (previously a short two-person dialogue
  could be merged into a single speaker).
- **No more `UNKNOWN` speakers**: `assign_word_speakers(...,
  fill_nearest=True)` (when supported) and the post-processor treats
  `UNKNOWN` as unlabeled and repairs it from neighbors.
- **Alignment resilience**: if no wav2vec2 alignment model exists for the
  detected language, the pipeline degrades to segment-level timestamps
  instead of failing.
