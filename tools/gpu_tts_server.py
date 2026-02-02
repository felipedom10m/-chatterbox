import base64
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
import tempfile

import numpy as np
import torch

from chatterbox.mtl_tts import ChatterboxMultilingualTTS


def _write_wav_bytes(sr, audio_np):
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767.0).astype(np.int16)

    import wave
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())
    return buffer.getvalue()


device = "cuda" if torch.cuda.is_available() else "cpu"
model = ChatterboxMultilingualTTS.from_pretrained(device)


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, code, payload):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_POST(self):
        if self.path != "/generate":
            return self._send_json(404, {"error": "not_found"})

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            return self._send_json(400, {"error": "invalid_json"})

        text = payload.get("text", "")
        audio_b64 = payload.get("audio_prompt_base64")
        if not text or not audio_b64:
            return self._send_json(400, {"error": "missing_text_or_audio"})

        language_id = payload.get("language_id", "pt")
        exaggeration = float(payload.get("exaggeration", 0.5))
        temperature = float(payload.get("temperature", 0.8))
        cfg_weight = float(payload.get("cfg_weight", 0.5))
        min_p = float(payload.get("min_p", 0.05))
        top_p = float(payload.get("top_p", 1.0))
        repetition_penalty = float(payload.get("repetition_penalty", 1.2))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(base64.b64decode(audio_b64))
            tmp_path = tmp.name

        try:
            model.prepare_conditionals(tmp_path, exaggeration=exaggeration)
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
            audio_np = wav.squeeze(0).detach().cpu().numpy()
            wav_bytes = _write_wav_bytes(model.sr, audio_np)
            audio_out = base64.b64encode(wav_bytes).decode("utf-8")
            return self._send_json(200, {"audio_wav_base64": audio_out})
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


if __name__ == "__main__":
    host = os.environ.get("GPU_TTS_HOST", "0.0.0.0")
    port = int(os.environ.get("GPU_TTS_PORT", "8000"))
    server = HTTPServer((host, port), Handler)
    print(f"GPU TTS server running on {host}:{port}")
    server.serve_forever()
