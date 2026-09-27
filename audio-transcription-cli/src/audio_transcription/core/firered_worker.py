"""FireRedASR2S worker — runs inside the dedicated firered Python environment.

This script is executed as a subprocess by FireRedBackend (which lives in
the main project environment). It loads the FireRedASR2 all-in-one system
(VAD + LID + ASR + Punc) once, transcribes one 16 kHz WAV file, and writes
the result as JSON.

Usage:
    python firered_worker.py --wav input.wav --out result.json \
        --models-root /path/to/pretrained_models [--asr-type llm] [--cpu]

Output JSON:
    {
        "text": str,
        "sentences": [{"start_ms", "end_ms", "text", "lang"?}, ...],
        "words": [{"start_ms", "end_ms", "text"}, ...]   # only when available
    }
"""

import argparse
import importlib
import json
import sys
import types


def _stub_torch_distributed() -> None:
    """Work around TheRock ROCm Windows PyTorch builds lacking c10d.

    Those builds ship no ``torch._C._distributed_c10d``, but transformers
    (4.51.x) imports ``torch.distributed.tensor`` unconditionally at module
    import time. The symbols are only used on distributed code paths we never
    hit, so stub the submodules that chain to the missing extension.
    """
    for name in (
        "torch.distributed.tensor",
        "torch.distributed.device_mesh",
        "torch.distributed._functional_collectives",
    ):
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as e:
            if "_distributed_c10d" not in str(e):
                raise
            sys.modules[name] = types.ModuleType(name)


_stub_torch_distributed()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--models-root", required=True)
    parser.add_argument("--asr-type", default="llm", choices=["aed", "llm"])
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    from fireredasr2s import FireRedAsr2System, FireRedAsr2SystemConfig
    from fireredasr2s.fireredasr2 import FireRedAsr2Config
    from fireredasr2s.fireredlid import FireRedLidConfig
    from fireredasr2s.fireredpunc import FireRedPuncConfig
    from fireredasr2s.fireredvad import FireRedVadConfig

    use_gpu = not args.cpu
    root = args.models_root.rstrip("/\\")

    vad_config = FireRedVadConfig(use_gpu=False)  # VAD is tiny; CPU is fine
    lid_config = FireRedLidConfig(use_gpu=use_gpu, use_half=False)
    punc_config = FireRedPuncConfig(use_gpu=use_gpu)

    if args.asr_type == "llm":
        asr_config = FireRedAsr2Config(
            use_gpu=use_gpu,
            decode_min_len=0,
            repetition_penalty=3.0,
            llm_length_penalty=1.0,
            temperature=1.0,
        )
        asr_dir = f"{root}/FireRedASR2-LLM"
    else:
        asr_config = FireRedAsr2Config(
            use_gpu=use_gpu,
            use_half=False,
            beam_size=3,
            nbest=1,
            decode_max_len=0,
            softmax_smoothing=1.25,
            aed_length_penalty=0.6,
            eos_penalty=1.0,
            return_timestamp=True,
        )
        asr_dir = f"{root}/FireRedASR2-AED"

    system_config = FireRedAsr2SystemConfig(
        vad_model_dir=f"{root}/FireRedVAD/VAD",
        lid_model_dir=f"{root}/FireRedLID",
        asr_type=args.asr_type,
        asr_model_dir=asr_dir,
        punc_model_dir=f"{root}/FireRedPunc",
        vad_config=vad_config,
        lid_config=lid_config,
        asr_config=asr_config,
        punc_config=punc_config,
        enable_vad=True,
        enable_lid=True,
        enable_punc=True,
    )

    system = FireRedAsr2System(system_config)
    result = system.process(args.wav, uttid="input")

    payload = {
        "text": result.get("text", ""),
        "sentences": result.get("sentences", []),
        "words": result.get("words", []),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
