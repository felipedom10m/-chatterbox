import random
import numpy as np
import torch
import gradio as gr

from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_model():
    model = ChatterboxMultilingualTTS.from_pretrained(DEVICE)
    return model


def generate(
    model,
    text,
    language_id,
    audio_prompt_path,
    exaggeration,
    temperature,
    seed_num,
    cfgw,
    min_p,
    top_p,
    repetition_penalty,
):
    texto_tamanho = len(text or "")
    print("═══════════════════════════════════════════")
    print("🎙️ GERAÇÃO INICIADA (Chatterbox)")
    print(f"🧠 Device: {DEVICE}")
    print(f"🗣️ Idioma: {language_id}")
    print(f"🔤 Tamanho do texto: {texto_tamanho} caracteres")
    print(f"🎧 Audio prompt: {'SIM' if audio_prompt_path else 'NÃO'}")
    print("═══════════════════════════════════════════")

    if model is None:
        model = ChatterboxMultilingualTTS.from_pretrained(DEVICE)

    if seed_num != 0:
        set_seed(int(seed_num))

    try:
        wav = model.generate(
            text,
            language_id=language_id,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfgw,
            min_p=min_p,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        print("✅ GERAÇÃO FINALIZADA COM SUCESSO")
        print("═══════════════════════════════════════════")
        return (model.sr, wav.squeeze(0).numpy())
    except Exception as erro:
        print("❌ ERRO NA GERAÇÃO")
        print(f"🧯 Motivo: {erro}")
        print("═══════════════════════════════════════════")
        raise


LANG_CHOICES = [(f"{name} ({code})", code) for code, name in SUPPORTED_LANGUAGES.items()]

with gr.Blocks() as demo:
    model_state = gr.State(None)

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Digite aqui o texto em português...",
                label="Texto para sintetizar (máx 300 caracteres)",
                max_lines=5,
                max_length=300,
            )

            language = gr.Dropdown(
                choices=LANG_CHOICES,
                value="pt",
                label="Idioma do texto",
            )

            ref_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Reference Audio File",
                value=None,
            )
            exaggeration = gr.Slider(
                0.25,
                2,
                step=.05,
                label="Exaggeration (Neutral = 0.5, extreme values can be unstable)",
                value=.5,
            )
            cfg_weight = gr.Slider(0.0, 1, step=.05, label="CFG/Pace", value=0.5)

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="temperature", value=.8)
                min_p = gr.Slider(
                    0.00,
                    1.00,
                    step=0.01,
                    label="min_p || Newer Sampler. Recommend 0.02 > 0.1. Handles Higher Temperatures better. 0.00 Disables",
                    value=0.05,
                )
                top_p = gr.Slider(
                    0.00,
                    1.00,
                    step=0.01,
                    label="top_p || Original Sampler. 1.0 Disables(recommended). Original 0.8",
                    value=1.00,
                )
                repetition_penalty = gr.Slider(1.00, 2.00, step=0.1, label="repetition_penalty", value=1.2)

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")

    demo.load(fn=load_model, inputs=[], outputs=model_state)

    run_btn.click(
        fn=generate,
        inputs=[
            model_state,
            text,
            language,
            ref_wav,
            exaggeration,
            temp,
            seed_num,
            cfg_weight,
            min_p,
            top_p,
            repetition_penalty,
        ],
        outputs=audio_output,
    )

if __name__ == "__main__":
    demo.queue(
        status_update_rate=3,
        max_size=50,
        default_concurrency_limit=1,
    ).launch(share=False, show_error=True)
