# 🎹 AI Piano Music Generation and Style Selection System

> AI piano music generation system with style-based selection and automatic evaluation.

---

## 🚀 Overview

This project implements an AI-based piano music generation system with style-based selection.

Instead of directly relying on the generative model, the system improves output quality through a structured pipeline:

---

## 💡 Key Idea

> 🎯 Generate multiple candidates, evaluate them, and select the best one.

Rather than modifying the generation model, this system improves results through a **generate–evaluate–select pipeline**, ensuring higher consistency and quality.

---

## 🧠 System Pipeline

### 🔹 Data Preprocessing (`split_audio.py`)
- Split raw audio into fixed 30-second segments  
- Remove clips shorter than 20 seconds  
- Organize dataset into:
  - `target_style` (desired piano style)
  - `other_style` (contrast data)

---

### 🔹 Model Training (`train_style_classifier.py`)
- Extract MFCC features using `librosa`  
- Train a Random Forest classifier  
- Output model: `style_model.pkl`  

🧠 Acts as the system’s **style evaluator**

---

### 🔹 Candidate Generation (`main.py`)
- Use MusicGen to generate piano music  
- Controlled by:
  - 🎭 Emotion  
  - ⏱ Tempo  
- Generate multiple candidates per request  

---

### 🔹 Evaluation & Selection (`app.py`)
- Compute **style matching scores**  
- Select the highest-scoring candidate  
- Provide all candidates for comparison  

---

## 🎧 Data Processing

Raw piano tracks (3–4 minutes) are not suitable for training due to:

- ❌ Mixed musical structure  
- ❌ Changing emotions  
- ❌ Noisy feature representation  

### ✅ Solution: Audio Segmentation

All audio is split into **30-second segments**, which provides:

- ✔ Stable features  
- ✔ Consistent style representation  
- ✔ Improved classification accuracy  

---

## 🖥 Demo (Gradio Interface)

The system provides an interactive UI where users can:

- 🎛 Select emotion and tempo  
- 🎶 Generate multiple candidates  
- 📊 View style matching scores  
- 🔊 Listen to:
  - Best selected audio  
  - All generated candidates  

---

## ⚙️ Installation

```bash
pip install transformers torch librosa scikit-learn gradio pydub soundfile
```

---

## 🚀 Run

```bash
python app.py
```

Then open in your browser:

```text
http://127.0.0.1:7860
```

---

## 📂 Dataset

The dataset is not included due to file size.
However, the dataset can be reproduced using the provided preprocessing script.

### Reproduction Steps

1. Place raw audio files in:

```text
/raw_music/
```

2. Run:

```bash
python split_audio.py
```
> Note: The system is designed to work with any piano-style audio dataset.
