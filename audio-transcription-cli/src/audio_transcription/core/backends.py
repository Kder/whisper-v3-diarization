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


def create_backend(name: str = "pytorch", **kwargs: Any) -> TranscriptionBackend:
    """Factory for transcription backends.

    Args:
        name: "pytorch" (original HF pipeline) or "whispercpp" (NPU offload).
        **kwargs: forwarded to the backend constructor.
    """
    backends = {
        PyTorchBackend.name: PyTorchBackend,
        WhisperCppBackend.name: WhisperCppBackend,
    }
    if name not in backends:
        raise ValueError(
            f"Unknown transcription backend: {name!r}. "
            f"Available: {', '.join(backends)}"
        )
    logger.info("Creating transcription backend: %s", name)
    return backends[name](**kwargs)
