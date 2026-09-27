"""Pluggable transcription backends.

Provides a common interface so the transcription engine can be swapped
between the original PyTorch/HuggingFace implementation and external
engines such as whisper.cpp (with AMD Ryzen AI NPU offload via VitisAI).

This module is intentionally dependency-light: it must be importable
without torch/transformers so the whisper.cpp backend can be used (and
tested) on machines that only run the NPU stack.
"""

import json
import logging
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TranscriptionBackend(ABC):
    """Abstract transcription backend.

    All backends return a result dict compatible with the original
    WhisperTranscriber.transcribe() contract:

        {
            "success": bool,
            "text": str,                # full transcript
            "chunks": [                 # segment-level timestamps (may be [])
                {"start": float, "end": float, "text": str}
            ],
            "language": str,
            "model": str,
            "timestamp": str,
        }

    On failure: {"success": False, "error": str, "model": str}
    """

    name: str = "abstract"

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transcribe an audio file. See class docstring for the contract."""

    def cleanup(self) -> None:
        """Release any held resources. Default: nothing to release."""


class PyTorchBackend(TranscriptionBackend):
    """Original HuggingFace/PyTorch Whisper backend (CUDA/CPU).

    Thin lazy wrapper around the existing WhisperTranscriber so that
    torch/transformers are only imported when this backend is selected.
    """

    name = "pytorch"

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cpu",
        use_assistant: bool = False,
    ):
        from .transcription import WhisperTranscriber  # deferred: pulls in torch

        self._impl = WhisperTranscriber(
            model_size=model_size, device=device, use_assistant=use_assistant
        )
        self.model_size = model_size

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = self._impl.transcribe(audio_path=audio_path, language=language)
        result.setdefault("backend", self.name)
        return result

    def cleanup(self) -> None:
        self._impl.cleanup()


class WhisperCppBackend(TranscriptionBackend):
    """whisper.cpp backend with AMD Ryzen AI NPU (VitisAI) offload.

    Invokes a whisper.cpp `whisper-cli` binary built with
    -DWHISPER_VITISAI=1. The encoder runs on the XDNA2 NPU through the
    FlexML runtime (the matching `*-vitisai.rai` cache must sit next to
    the ggml model); the decoder runs on CPU. Segment-level timestamps
    are preserved via whisper.cpp's JSON output.

    Configuration is resolved in this order (first match wins):
        constructor arg > environment variable > built-in default

        exe_path      / WHISPER_CPP_EXE
        model_path    / WHISPER_CPP_MODEL
        flexmlrt_root / FLEXMLRT_ROOT   (Windows NPU runtime)
    """

    name = "whispercpp"

    DEFAULT_EXE = r"D:\Programs\whisper.cpp\build\bin\Release\whisper-cli.exe"
    DEFAULT_MODEL = r"D:\Programs\whisper.cpp\models\ggml-large-v3-turbo.bin"
    DEFAULT_FLEXMLRT = r"D:\Apps\flexmlrt"

    def __init__(
        self,
        exe_path: Optional[str] = None,
        model_path: Optional[str] = None,
        flexmlrt_root: Optional[str] = None,
        threads: int = 4,
        extra_args: Optional[List[str]] = None,
        timeout: int = 3600,
    ):
        self.exe_path = Path(
            exe_path or os.getenv("WHISPER_CPP_EXE") or self.DEFAULT_EXE
        )
        self.model_path = Path(
            model_path or os.getenv("WHISPER_CPP_MODEL") or self.DEFAULT_MODEL
        )
        self.flexmlrt_root = Path(
            flexmlrt_root or os.getenv("FLEXMLRT_ROOT") or self.DEFAULT_FLEXMLRT
        )
        self.threads = threads
        self.extra_args = extra_args or []
        self.timeout = timeout

        logger.info(
            "WhisperCppBackend initialized - exe: %s, model: %s",
            self.exe_path,
            self.model_path,
        )

    # ------------------------------------------------------------------ API

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        model_name = self.model_path.stem
        try:
            self._validate_installation()

            with tempfile.TemporaryDirectory(prefix="whispercpp_") as tmp:
                tmp_dir = Path(tmp)
                wav_path = self._to_16k_wav(audio_path, tmp_dir)
                out_base = tmp_dir / "result"

                cmd = [
                    str(self.exe_path),
                    "-m", str(self.model_path),
                    "-f", str(wav_path),
                    "-oj",                      # JSON output -> <out_base>.json
                    "-of", str(out_base),
                    "-t", str(self.threads),
                ]
                if language:
                    cmd += ["-l", language]
                cmd += self.extra_args

                logger.info("Running whisper.cpp: %s", " ".join(cmd))
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout,
                    env=self._build_env(),
                )

                log_output = (proc.stdout or "") + (proc.stderr or "")
                npu_offload = "Vitis AI encoder model loaded" in log_output
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"whisper-cli exited with code {proc.returncode}: "
                        f"{log_output[-2000:]}"
                    )
                if not npu_offload:
                    logger.warning(
                        "VitisAI encoder offload NOT detected in whisper.cpp log; "
                        "transcription ran on CPU only. Check flexmlrt setup and "
                        "the .rai cache next to the model."
                    )

                json_file = out_base.with_suffix(".json")
                if not json_file.exists():
                    raise RuntimeError(
                        f"whisper-cli did not produce {json_file}: "
                        f"{log_output[-2000:]}"
                    )
                segments, detected_language = self._parse_json(json_file)

            text = " ".join(seg["text"] for seg in segments).strip()
            return {
                "success": True,
                "text": text,
                "chunks": segments,
                "language": language or detected_language or "auto-detected",
                "model": model_name,
                "backend": self.name,
                "npu_offload": npu_offload,
                "timestamp": str(datetime.now()),
            }

        except Exception as e:
            logger.exception("whisper.cpp transcription failed for %s", audio_path)
            return {
                "success": False,
                "error": str(e),
                "model": model_name,
                "backend": self.name,
            }

    # ------------------------------------------------------------- internal

    def _validate_installation(self) -> None:
        if not self.exe_path.exists():
            raise FileNotFoundError(
                f"whisper-cli not found at {self.exe_path}. "
                "Build whisper.cpp with -DWHISPER_VITISAI=1 or set WHISPER_CPP_EXE."
            )
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ggml model not found at {self.model_path}. "
                "Set WHISPER_CPP_MODEL to a valid model file."
            )
        rai = self.model_path.with_name(
            self.model_path.stem + "-encoder-vitisai.rai"
        )
        if not rai.exists():
            logger.warning(
                "VitisAI encoder cache %s not found; NPU offload will not happen.",
                rai,
            )

    def _build_env(self) -> Dict[str, str]:
        """Environment for the subprocess, adding the FlexML runtime (NPU)."""
        env = os.environ.copy()
        if self.flexmlrt_root.exists():
            lib_dir = self.flexmlrt_root / "lib"
            cmake_dir = self.flexmlrt_root / "share" / "cmake" / "FlexmlRT"
            if lib_dir.exists():
                env["PATH"] = str(lib_dir) + os.pathsep + env.get("PATH", "")
            if cmake_dir.exists():
                env.setdefault("FlexmlRT_DIR", str(cmake_dir))
        return env

    @staticmethod
    def _to_16k_wav(audio_path: Path, tmp_dir: Path) -> Path:
        """Convert input audio to 16 kHz mono WAV for whisper.cpp.

        Uses librosa/soundfile (already project dependencies). Falls back
        to passing the original file through if conversion is unavailable.
        """
        try:
            import librosa
            import soundfile as sf

            data, _ = librosa.load(str(audio_path), sr=16000, mono=True)
            wav_path = tmp_dir / "input_16k.wav"
            sf.write(str(wav_path), data, 16000, subtype="PCM_16")
            return wav_path
        except Exception as e:
            logger.warning(
                "16kHz WAV conversion failed (%s); passing original file to "
                "whisper-cli and relying on its built-in decoder.",
                e,
            )
            return audio_path

    @staticmethod
    def _parse_json(json_file: Path) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """Parse whisper.cpp -oj output into the segment contract."""
        with open(json_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        segments: List[Dict[str, Any]] = []
        for item in payload.get("transcription", []):
            offsets = item.get("offsets", {})
            text = (item.get("text") or "").strip()
            if not text:
                continue
            segments.append(
                {
                    "start": offsets.get("from", 0) / 1000.0,
                    "end": offsets.get("to", 0) / 1000.0,
                    "text": text,
                }
            )

        detected_language = (payload.get("result") or {}).get("language")
        return segments, detected_language


class FunASRBackend(TranscriptionBackend):
    """FunASR (Alibaba Tongyi Lab) backend — Chinese-optimized ASR with
    optional built-in CAM++ speaker diarization.

    A single ``generate()`` call chains VAD (fsmn-vad) + ASR (paraformer-zh
    or SenseVoice) + punctuation (ct-punc) + speaker embedding clustering
    (CAM++), returning sentence-level text, timestamps AND speaker labels —
    no HuggingFace token or gated-model acceptance required (models download
    from ModelScope).

    When ``spk_model`` is set, ``transcribe()`` chunks carry a ``speaker``
    field and ``diarize()`` can produce the full diarization result without
    touching WhisperX/pyannote at all.
    """

    name = "funasr"

    def __init__(
        self,
        model: Optional[str] = None,
        vad_model: Optional[str] = "fsmn-vad",
        punc_model: Optional[str] = "ct-punc",
        spk_model: Optional[str] = "cam++",
        device: str = "cpu",
        batch_size_s: int = 300,
    ):
        # FUNASR_MODEL env var selects e.g. the larger Fun-ASR-Nano-2512
        # (800M, Tongyi Lab) instead of the default paraformer-zh (220M).
        self.model = model or os.getenv("FUNASR_MODEL") or "paraformer-zh"
        self.vad_model = vad_model
        self.punc_model = punc_model
        self.spk_model = spk_model
        self.device = device
        self.batch_size_s = batch_size_s
        self._impl = None  # lazy AutoModel
        if self._is_fun_asr_nano:
            # No CAM++ chain for this model family — callers check
            # spk_model to decide whether built-in diarization exists.
            self.spk_model = None

        logger.info(
            "FunASRBackend initialized - model: %s, spk: %s, device: %s",
            self.model, self.spk_model, device,
        )

    # ------------------------------------------------------------------ API

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            self._load()
            result = self._run(audio_path, language)

            segments = self._extract_segments(result)
            text = " ".join(seg["text"] for seg in segments).strip()
            if not text:
                text = (result.get("text") or "").strip()

            has_speakers = any("speaker" in seg for seg in segments)
            return {
                "success": True,
                "text": text,
                "chunks": segments,
                "language": language or "zh",
                "model": self.model,
                "backend": self.name,
                "has_speakers": has_speakers,
                "timestamp": str(datetime.now()),
            }

        except Exception as e:
            logger.exception("FunASR transcription failed for %s", audio_path)
            return {
                "success": False,
                "error": str(e),
                "model": self.model,
                "backend": self.name,
            }

    def diarize(
        self,
        audio_path: Path,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Full diarization result from the built-in CAM++ pipeline.

        Returns the same contract as WhisperXDiarizer.diarize() so the rest
        of the application (file saving, GUI) works unchanged.
        """
        if not self.spk_model:
            return {
                "success": False,
                "error": "FunASRBackend was created without spk_model; "
                         "speaker diarization is unavailable.",
            }

        transcript = self.transcribe(audio_path=audio_path, language=language)
        if not transcript["success"]:
            return transcript

        segments = transcript["chunks"]
        if not any("speaker" in seg for seg in segments):
            return {
                "success": False,
                "error": "FunASR result contains no speaker labels "
                         "(sentence_info missing).",
            }

        speakers = sorted({seg["speaker"] for seg in segments})
        formatted = self._format_transcript(segments)

        return {
            "success": True,
            "segments": segments,
            "num_speakers": len(speakers),
            "speakers": speakers,
            "language": transcript["language"],
            "formatted_transcript": formatted,
            "model": transcript["model"],
            "backend": self.name,
            "diarization_method": f"funasr-{self.spk_model}",
        }

    def cleanup(self) -> None:
        if self._impl is not None:
            del self._impl
            self._impl = None
            logger.info("FunASR model cleaned up")

    # ------------------------------------------------------------- internal

    @property
    def _is_fun_asr_nano(self) -> bool:
        """Fun-ASR-Nano (800M LLM-based ASR) — punctuates natively and does
        not chain ct-punc / CAM++ the way paraformer-zh does."""
        low = self.model.lower()
        return "fun-asr" in low or "fun_asr" in low

    def _load(self) -> None:
        if self._impl is not None:
            return
        from funasr import AutoModel  # deferred: heavy import

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "device": self.device,
            "disable_update": True,
        }
        if self.vad_model:
            kwargs["vad_model"] = self.vad_model
            if self._is_fun_asr_nano:
                kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}
        if self._is_fun_asr_nano:
            # LLM-based: outputs punctuation itself; no CAM++ speaker chain
            if self.punc_model or self.spk_model:
                logger.info(
                    "Fun-ASR-Nano does not chain punc/spk models; "
                    "diarization falls back to pyannote"
                )
        else:
            if self.punc_model:
                kwargs["punc_model"] = self.punc_model
            if self.spk_model:
                kwargs["spk_model"] = self.spk_model

        logger.info("Loading FunASR pipeline: %s", kwargs)
        self._impl = AutoModel(**kwargs)
        logger.info("FunASR pipeline loaded")

    def _run(self, audio_path: Path, language: Optional[str]) -> Dict[str, Any]:
        gen_kwargs: Dict[str, Any] = {
            "input": str(audio_path),
            "batch_size_s": self.batch_size_s,
        }
        low = self.model.lower()
        # SenseVoice-family models accept language/itn; paraformer-zh does not
        if "sensevoice" in low:
            gen_kwargs["language"] = language or "auto"
            gen_kwargs["use_itn"] = True
        elif self._is_fun_asr_nano:
            gen_kwargs.pop("batch_size_s")
            gen_kwargs["batch_size"] = 1
            lang_map = {"zh": "中文", "en": "英文", "ja": "日文"}
            gen_kwargs["language"] = lang_map.get((language or "zh")[:2], "中文")
            gen_kwargs["itn"] = True

        res = self._impl.generate(**gen_kwargs)
        if not res:
            raise RuntimeError("FunASR returned an empty result")
        return res[0]

    @staticmethod
    def _extract_segments(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Map FunASR sentence_info to the segment contract.

        sentence_info entries: {"start": ms, "end": ms, "sentence"|"text",
        "spk": int, ...}
        """
        segments: List[Dict[str, Any]] = []
        for item in result.get("sentence_info") or []:
            text = (item.get("sentence") or item.get("text") or "").strip()
            if not text:
                continue
            seg: Dict[str, Any] = {
                "start": item.get("start", 0) / 1000.0,
                "end": item.get("end", 0) / 1000.0,
                "text": text,
            }
            if "spk" in item:
                seg["speaker"] = f"SPEAKER_{int(item['spk']):02d}"
            segments.append(seg)
        if segments:
            return segments
        # Fun-ASR-Nano: no sentence_info, but token-level timestamps
        # ([{"token", "start_time", "end_time", "score"}, ...], seconds).
        # Rebuild sentence segments by splitting on sentence punctuation.
        tokens = result.get("timestamps") or []
        if tokens and isinstance(tokens[0], dict) and "token" in tokens[0]:
            buf_text: List[str] = []
            seg_start: Optional[float] = None
            seg_end: Optional[float] = None
            for tok in tokens:
                piece = tok.get("token", "")
                if seg_start is None and piece.strip():
                    seg_start = float(tok.get("start_time", 0.0))
                buf_text.append(piece)
                if piece.strip():
                    seg_end = float(tok.get("end_time", seg_end or 0.0))
                if piece.strip() in "。！？；!?;":
                    text = "".join(buf_text).strip()
                    if text:
                        segments.append({
                            "start": seg_start or 0.0,
                            "end": seg_end or seg_start or 0.0,
                            "text": text,
                        })
                    buf_text, seg_start, seg_end = [], None, None
            text = "".join(buf_text).strip()
            if text:
                segments.append({
                    "start": seg_start or 0.0,
                    "end": seg_end or seg_start or 0.0,
                    "text": text,
                })
        return segments

    @staticmethod
    def _format_transcript(segments: List[Dict[str, Any]]) -> str:
        lines = []
        for seg in segments:
            minutes = int(seg["start"] // 60)
            secs = int(seg["start"] % 60)
            speaker = seg.get("speaker", "UNKNOWN")
            lines.append(f"[{minutes:02d}:{secs:02d}] {speaker}: {seg['text']}")
        return "\n".join(lines)


class FireRedBackend(TranscriptionBackend):
    """FireRedASR2S backend (Xiaohongshu) — top-accuracy Chinese ASR.

    Runs the FireRedASR2 all-in-one system (FireRedVAD + FireRedLID +
    FireRedASR2 + FireRedPunc) in a DEDICATED Python environment via a
    subprocess worker, because it needs its own PyTorch build (e.g. a ROCm
    gfx1151 build for AMD iGPU inference) that would conflict with the main
    project environment.

    Configuration (constructor arg > env var > default):
        python_path  / FIRERED_PYTHON       Python of the firered venv
        models_root  / FIRERED_MODELS_ROOT  dir containing FireRedASR2-LLM,
                                            FireRedVAD, FireRedLID, FireRedPunc
    """

    name = "firered"

    DEFAULT_PYTHON = (
        r"C:\Users\llf21\Documents\kimi\tasks\2026-09-23\22-51-06-22cd18c8"
        r"\firered-venv\Scripts\python.exe"
    )
    DEFAULT_MODELS_ROOT = (
        r"C:\Users\llf21\Documents\kimi\tasks\2026-09-23\22-51-06-22cd18c8"
        r"\FireRedASR2S-main\pretrained_models"
    )

    def __init__(
        self,
        python_path: Optional[str] = None,
        models_root: Optional[str] = None,
        asr_type: str = "llm",
        use_gpu: bool = True,
        timeout: int = 7200,
    ):
        self.python_path = Path(
            python_path or os.getenv("FIRERED_PYTHON") or self.DEFAULT_PYTHON
        )
        self.models_root = Path(
            models_root or os.getenv("FIRERED_MODELS_ROOT") or self.DEFAULT_MODELS_ROOT
        )
        self.asr_type = asr_type
        self.use_gpu = use_gpu
        self.timeout = timeout
        self.worker_path = Path(__file__).parent / "firered_worker.py"

        logger.info(
            "FireRedBackend initialized - python: %s, models: %s, asr: %s, gpu: %s",
            self.python_path, self.models_root, asr_type, use_gpu,
        )

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        model_name = f"FireRedASR2-{self.asr_type.upper()}"
        try:
            if not self.python_path.exists():
                raise FileNotFoundError(
                    f"FireRed Python not found at {self.python_path}. "
                    "Set FIRERED_PYTHON to the dedicated venv interpreter."
                )
            if not self.worker_path.exists():
                raise FileNotFoundError(f"Worker script missing: {self.worker_path}")

            with tempfile.TemporaryDirectory(prefix="firered_") as tmp:
                tmp_dir = Path(tmp)
                wav_path = WhisperCppBackend._to_16k_wav(audio_path, tmp_dir)
                out_json = tmp_dir / "result.json"

                cmd = [
                    str(self.python_path),
                    str(self.worker_path),
                    "--wav", str(wav_path),
                    "--out", str(out_json),
                    "--models-root", str(self.models_root),
                    "--asr-type", self.asr_type,
                ]
                if not self.use_gpu:
                    cmd.append("--cpu")

                logger.info("Running FireRedASR2S worker: %s", " ".join(cmd))
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout,
                )
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"FireRed worker exited with code {proc.returncode}: "
                        f"{(proc.stderr or proc.stdout or '')[-2000:]}"
                    )
                if not out_json.exists():
                    raise RuntimeError("FireRed worker produced no output file")

                with open(out_json, "r", encoding="utf-8") as f:
                    payload = json.load(f)

            segments: List[Dict[str, Any]] = []
            for sent in payload.get("sentences", []):
                text = (sent.get("text") or "").strip()
                if not text:
                    continue
                segments.append(
                    {
                        "start": sent.get("start_ms", 0) / 1000.0,
                        "end": sent.get("end_ms", 0) / 1000.0,
                        "text": text,
                    }
                )

            text = " ".join(seg["text"] for seg in segments).strip()
            if not text:
                text = (payload.get("text") or "").strip()

            return {
                "success": True,
                "text": text,
                "chunks": segments,
                "language": language or "zh",
                "model": model_name,
                "backend": self.name,
                "timestamp": str(datetime.now()),
            }

        except Exception as e:
            logger.exception("FireRedASR2S transcription failed for %s", audio_path)
            return {
                "success": False,
                "error": str(e),
                "model": model_name,
                "backend": self.name,
            }


def create_backend(name: str = "pytorch", **kwargs: Any) -> TranscriptionBackend:
    """Factory for transcription backends.

    Args:
        name: "pytorch" (original HF pipeline), "whispercpp" (NPU offload),
            "funasr" (Chinese-optimized, built-in CAM++ diarization) or
            "firered" (FireRedASR2S, top-accuracy Chinese ASR, GPU via ROCm).
        **kwargs: forwarded to the backend constructor.
    """
    backends = {
        PyTorchBackend.name: PyTorchBackend,
        WhisperCppBackend.name: WhisperCppBackend,
        FunASRBackend.name: FunASRBackend,
        FireRedBackend.name: FireRedBackend,
    }
    if name not in backends:
        raise ValueError(
            f"Unknown transcription backend: {name!r}. "
            f"Available: {', '.join(backends)}"
        )
    logger.info("Creating transcription backend: %s", name)
    return backends[name](**kwargs)
