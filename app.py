from transformers import pipeline
import joblib
import librosa
import numpy as np
import gradio as gr

print("Loading MusicGen...")
generator = pipeline("text-to-audio", model="facebook/musicgen-small")

print("Loading style classifier...")
style_model = joblib.load("style_model.pkl")


def extract_features(audio, sr):
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13), axis=1)
    return mfcc.reshape(1, -1)


def normalize_audio(audio):
    audio = np.array(audio).squeeze()

    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    return audio.astype(np.float32)


def generate_music(emotion, tempo, candidates):
    prompt = (
        f"{emotion} anime-style solo piano music, "
        f"{tempo} tempo, emotional melody, Japanese animation soundtrack style, "
        f"soft and expressive piano arrangement, cinematic, no drums, no vocals"
    )

    best_score = -1
    best_audio = None
    best_sr = None

    candidate_audios = []
    scores_text = f"Prompt: {prompt}\n\n"

    for i in range(int(candidates)):
        print(f"Generating candidate {i + 1}...")

        music = generator(prompt, forward_params={"do_sample": True})

        audio = normalize_audio(music["audio"])
        sr = music["sampling_rate"]

        features = extract_features(audio, sr)
        score = style_model.predict_proba(features)[0][1]

        scores_text += f"Candidate {i + 1} Style Matching Score: {score:.3f}\n"

        candidate_audios.append(audio)

        if score > best_score:
            best_score = score
            best_audio = audio
            best_sr = sr

    silence = np.zeros(int(best_sr * 1.0), dtype=np.float32)

    all_candidates_audio = []
    for i, audio in enumerate(candidate_audios):
        all_candidates_audio.append(audio)
        all_candidates_audio.append(silence)

    all_candidates_audio = np.concatenate(all_candidates_audio).astype(np.float32)

    result = f"Best Matching Score: {best_score:.3f}\n\n{scores_text}"

    return (best_sr, best_audio), result, (best_sr, all_candidates_audio)


demo = gr.Interface(
    fn=generate_music,
    inputs=[
        gr.Dropdown(
            ["sad", "romantic", "peaceful", "dark", "emotional", "nostalgic"],
            label="Emotion"
        ),
        gr.Dropdown(
            ["slow", "medium"],
            label="Tempo"
        ),
        gr.Slider(
            minimum=3,
            maximum=8,
            value=5,
            step=1,
            label="Number of Candidates"
        )
    ],
    outputs=[
        gr.Audio(label="Selected Best Piano Music"),
        gr.Textbox(label="Style Matching Results"),
        gr.Audio(label="All Generated Candidates")
    ],
    title="AI Piano Music Generation and Style Selection System",
    description=(
        "The system generates multiple anime-style piano music candidates, "
        "uses a self-trained style classifier to evaluate them, "
        "and selects the candidate that best matches the target style."
    )
)

demo.launch()