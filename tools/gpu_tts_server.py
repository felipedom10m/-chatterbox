import base64
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
import tempfile
import time
import uuid

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
voice_sessions = {}
VOICE_SESSION_TTL_SEC = int(os.environ.get("GPU_TTS_SESSION_TTL_SEC", "7200"))
VOICE_SESSION_MAX = int(os.environ.get("GPU_TTS_SESSION_MAX", "64"))


def _cleanup_voice_sessions():
    now = time.time()
    expiradas = [
        session_id
        for session_id, info in voice_sessions.items()
        if now - float(info.get("updated_at", now)) > VOICE_SESSION_TTL_SEC
    ]
    for session_id in expiradas:
        voice_sessions.pop(session_id, None)

    while len(voice_sessions) > VOICE_SESSION_MAX:
        session_id_mais_antiga = min(
            voice_sessions.items(),
            key=lambda item: float(item[1].get("updated_at", now))
        )[0]
        voice_sessions.pop(session_id_mais_antiga, None)


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, code, payload):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def _read_payload(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8")), None
        except Exception:
            return None, {"error": "invalid_json"}

    def _prepare_conditionals_from_b64(self, audio_b64, exaggeration):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(base64.b64decode(audio_b64))
            tmp_path = tmp.name
        try:
            model.prepare_conditionals(tmp_path, exaggeration=exaggeration)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def do_POST(self):
        _cleanup_voice_sessions()

        payload, erro_payload = self._read_payload()
        if erro_payload:
            return self._send_json(400, erro_payload)

        if self.path == "/voice-session":
            audio_b64 = payload.get("audio_prompt_base64")
            if not audio_b64:
                return self._send_json(400, {"error": "missing_audio_prompt"})
            try:
                exaggeration = float(payload.get("exaggeration", 0.5))
                self._prepare_conditionals_from_b64(audio_b64, exaggeration=exaggeration)
            except Exception as erro:
                return self._send_json(500, {"error": "voice_session_prepare_failed", "detail": str(erro)})

            voice_session_id = uuid.uuid4().hex
            voice_sessions[voice_session_id] = {
                "conds": model.conds,
                "created_at": time.time(),
                "updated_at": time.time(),
            }
            print(
                f"[voice-session] criada {voice_session_id[:8]} | "
                f"ativas={len(voice_sessions)} | ttl={VOICE_SESSION_TTL_SEC}s"
            )
            return self._send_json(200, {"voice_session_id": voice_session_id})

        if self.path != "/generate":
            return self._send_json(404, {"error": "not_found"})

        text = payload.get("text", "")
        if not text:
            return self._send_json(400, {"error": "missing_text"})

        voice_session_id = str(payload.get("voice_session_id", "")).strip()
        language_id = payload.get("language_id", "pt")
        exaggeration = float(payload.get("exaggeration", 0.5))
        temperature = float(payload.get("temperature", 0.8))
        cfg_weight = float(payload.get("cfg_weight", 0.5))
        min_p = float(payload.get("min_p", 0.05))
        top_p = float(payload.get("top_p", 1.0))
        repetition_penalty = float(payload.get("repetition_penalty", 1.2))

        if voice_session_id:
            sessao = voice_sessions.get(voice_session_id)
            if not sessao:
                return self._send_json(400, {"error": "invalid_voice_session"})
            model.conds = sessao["conds"]
            sessao["updated_at"] = time.time()
            print(f"[generate] sessão {voice_session_id[:8]} | texto={len(text)} chars")
        else:
            audio_b64 = payload.get("audio_prompt_base64")
            if not audio_b64:
                return self._send_json(400, {"error": "missing_text_or_audio"})
            try:
                self._prepare_conditionals_from_b64(audio_b64, exaggeration=exaggeration)
            except Exception as erro:
                return self._send_json(500, {"error": "inline_prepare_failed", "detail": str(erro)})
            print(f"[generate] modo_inline | texto={len(text)} chars")

        try:
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
            if voice_session_id and voice_session_id in voice_sessions:
                voice_sessions[voice_session_id]["conds"] = model.conds
                voice_sessions[voice_session_id]["updated_at"] = time.time()

            audio_np = wav.squeeze(0).detach().cpu().numpy()
            wav_bytes = _write_wav_bytes(model.sr, audio_np)
            audio_out = base64.b64encode(wav_bytes).decode("utf-8")
            return self._send_json(200, {"audio_wav_base64": audio_out})
        except Exception as erro:
            return self._send_json(500, {"error": "generate_failed", "detail": str(erro)})


if __name__ == "__main__":
    host = os.environ.get("GPU_TTS_HOST", "0.0.0.0")
    port = int(os.environ.get("GPU_TTS_PORT", "8000"))
    server = HTTPServer((host, port), Handler)
    print(f"GPU TTS server running on {host}:{port}")
    server.serve_forever()
