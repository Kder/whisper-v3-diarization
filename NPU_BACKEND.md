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

---

# Alternative: FunASR Backend (Chinese-optimized, built-in diarization)

For **pure-Chinese** meetings, the `funasr` backend is often the better
choice: Alibaba's FunASR chains VAD (fsmn-vad) + ASR (paraformer-zh) +
punctuation (ct-punc) + **CAM++ speaker diarization** in a single
`generate()` call — no HuggingFace token, no gated models (weights download
from ModelScope), and no WhisperX/pyannote involvement at all.

```bash
pip install funasr modelscope

# One call: transcription + punctuation + speaker labels
audio-transcription diarize meeting.mp3 --backend funasr
```

## Measured comparison (same 5-minute real-world Chinese meeting, CPU-only)

| | `whispercpp` (NPU) + pyannote | `funasr` (CAM++) |
| --- | --- | --- |
| End-to-end time | ~220 s | **~17 s** (RTF ≈ 0.06) |
| Transcription | Whisper large-v3-turbo | paraformer-zh |
| Chinese output | occasional traditional-character bias | simplified, with punctuation |
| Speakers found | 2 | 5 |
| HF token / gated models | required (pyannote) | **not required** |

Notes:

- FunASR's Paraformer is non-autoregressive and extremely fast on CPU; the
  NPU's speed advantage does not matter for this workload class.
- CAM++ and pyannote can disagree on speaker count (CAM++ tends to find
  more speakers); neither is ground truth — spot-check on your own audio.
- The `funasr` backend runs on any machine (no NPU needed); use
  `whispercpp` when you want Whisper's multilingual coverage or already
  have the NPU stack set up.
- SenseVoice (`model="iic/SenseVoiceSmall"`) is also supported as the ASR
  model and adds emotion / audio-event tags.

## FireRedASR2S backend (top-accuracy Chinese ASR on AMD iGPU via ROCm)

`--backend firered` runs Xiaohongshu's FireRedASR2 all-in-one system
(FireRedVAD + FireRedLID + FireRedASR2 + FireRedPunc). The LLM variant
(FireRedASR2-LLM, 8.3B: Qwen2-7B + speech encoder) reports the lowest
publicly available Chinese CER (2.89% on AISHELL-1 test). Because it needs
its own PyTorch build (ROCm gfx1151 nightly for Strix Halo iGPUs), it runs
in a **dedicated venv** via a subprocess worker
(`core/firered_worker.py`); diarization then reuses the transcript through
pyannote like the `whispercpp` path.

### Setup (Windows, Strix Halo gfx1151)

```bash
# 1. Dedicated venv with TheRock ROCm PyTorch (gfx1151 nightly)
uv venv firered-venv
firered-venv\Scripts\python -m pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ torch
firered-venv\Scripts\python -m pip install --no-deps <FireRedASR2S source dir>
firered-venv\Scripts\python -m pip install transformers==4.51.3 soundfile librosa ...

# 2. Models (ModelScope, ~19 GB total)
modelscope download --model FireRedTeam/FireRedASR2-LLM --local_dir pretrained_models/FireRedASR2-LLM
modelscope download --model FireRedTeam/FireRedVAD   --local_dir pretrained_models/FireRedVAD
modelscope download --model FireRedTeam/FireRedLID   --local_dir pretrained_models/FireRedLID
modelscope download --model FireRedTeam/FireRedPunc  --local_dir pretrained_models/FireRedPunc

# 3. One-line upstream patch: fireredasr2system.py accesses
#    asr_result["confidence"] which the LLM path never returns — change the
#    three occurrences to asr_result.get("confidence", 0).

# 4. Point the backend at the venv (or edit FireRedBackend.DEFAULT_*)
set FIRERED_PYTHON=<abs path>\firered-venv\Scripts\python.exe
set FIRERED_MODELS_ROOT=<abs path>\pretrained_models

audio-transcription diarize meeting.mp3 --backend firered --language zh
```

The worker also stubs `torch.distributed.tensor` at import time: TheRock
Windows builds ship without `torch._C._distributed_c10d`, which
transformers 4.51.x imports unconditionally (the symbols are never used on
single-GPU inference paths).

### Measured comparison (same 5-minute real-world Chinese meeting)

| | `funasr` (CPU) | `firered` LLM 8.3B (iGPU ROCm) | `whispercpp` (NPU) + pyannote | `pytorch` large-v3 (CPU) |
| --- | --- | --- | --- | --- |
| End-to-end time | **~17 s** | ~436 s | ~220 s | hours |
| Transcription only | ~17 s | ~319 s (RTF ≈ 1.06) | ~98 s | — |
| Chinese accuracy | good | **best** (CER 2.89% claim) | good | good |
| Simplified + punctuation | yes | yes | occasional traditional bias | occasional traditional bias |
| Speakers found | 5 | 2 | 2 | 2 |
| HF token needed | **no** | yes (pyannote diarization) | yes | yes |
| Hardware | any CPU | AMD iGPU (ROCm) or CPU | Ryzen AI NPU | any |

Backend selection guide:

- **`funasr`** — default for Chinese meetings: fastest, no HF token,
  built-in diarization.
- **`firered`** — when Chinese transcription accuracy matters more than
  speed (formal minutes, downstream LLM summarization). AED variant
  (`asr_type="aed"`) is ~10x faster with slightly lower accuracy.
- **`whispercpp`** — multilingual coverage, or to offload work to the NPU
  and keep CPU/GPU free.
- **`pytorch`** — original pipeline, kept as the reference implementation.
