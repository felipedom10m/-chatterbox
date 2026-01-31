import json
import re
import sys

import numpy as np
import torch

from chatterbox.mtl_tts import ChatterboxMultilingualTTS


def _write_wav(path, sr, audio_np):
    # Normalize and write PCM16 WAV using stdlib
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767.0).astype(np.int16)

    import wave
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())


def main():
    raw = sys.stdin.read()
    if not raw:
        print("No input JSON received.", file=sys.stderr)
        sys.exit(1)

    payload = json.loads(raw)
    jobs = payload.get("jobs", [])
    if not jobs:
        print("No jobs provided.", file=sys.stderr)
        sys.exit(1)

    language_id = payload.get("language_id", "pt")
    audio_prompt_path = payload.get("audio_prompt_path")
    exaggeration = float(payload.get("exaggeration", 0.5))
    temperature = float(payload.get("temperature", 0.8))
    cfg_weight = float(payload.get("cfg_weight", 0.5))
    min_p = float(payload.get("min_p", 0.05))
    top_p = float(payload.get("top_p", 1.0))
    repetition_penalty = float(payload.get("repetition_penalty", 1.2))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxMultilingualTTS.from_pretrained(device)

    if audio_prompt_path:
        model.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)

    total = len(jobs)
    for idx, job in enumerate(jobs, start=1):
        text = job.get("text", "")
        out_path = job.get("out_path")
        prefix = job.get("log_prefix", f"[Parte {idx}/{total}]")
        if not out_path:
            print("Job missing out_path.", file=sys.stderr)
            sys.exit(1)

        print(f"{prefix}: iniciou", flush=True)

        class _LogFilter:
            def __init__(self):
                self.last_percent = -1

            def write(self, s):
                match = re.search(r"Sampling:\s+(\d+)%", s)
                if not match:
                    return
                percent = int(match.group(1))
                if percent % 5 == 0 and percent != self.last_percent:
                    print(f"{prefix}: {percent}%", flush=True)
                    self.last_percent = percent

            def flush(self):
                return

        old_stdout = sys.stdout
        sys.stdout = _LogFilter()

        wav = model.generate(
            text,
            language_id=language_id,
            audio_prompt_path=None,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight,
            min_p=min_p,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        sys.stdout = old_stdout

        audio_np = wav.squeeze(0).detach().cpu().numpy()
        _write_wav(out_path, model.sr, audio_np)

    print("OK")


if __name__ == "__main__":
    main()
