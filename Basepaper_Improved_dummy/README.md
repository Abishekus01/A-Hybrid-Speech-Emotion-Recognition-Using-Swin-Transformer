# 🎧 SwinTSER – Bilingual Speech Emotion Recognition  
### Using Shifted Window Transformer (Swin Transformer)

## 📌 Project Overview
This project implements a **Speech Emotion Recognition (SER)** system that predicts human emotions from speech audio.  
It is inspired by the research paper:

> **“SwinTSER: An Improved Bilingual Speech Emotion Recognition Using Shift Window Transformer”**

The system supports **bilingual audio (English + Tamil)** and uses a **deep learning Swin Transformer model** to classify emotions from audio features.

---

## 🎯 Objectives
- To recognize emotions from speech audio automatically
- To support **bilingual speech inputs**
- To apply **Transformer-based deep learning** instead of traditional CNN/RNN models (used MFCC)
- To provide a **web-based interface** for easy interaction

---

## 🧠 Key Idea (In Simple Terms)
1. Audio is uploaded via a web page  
2. Audio is converted into **Mel-Spectrograms** (audio → image-like representation)  
3. A **Swin Transformer** model learns emotion-related patterns  
4. The predicted emotion is displayed on the web interface  

---

## 🎭 Supported Emotions
Example emotion classes:
- Happy  
- Sad  
- Angry  
- Neutral  
- Fear  
- Disgust  
- Surprise  

(Exact labels can be configured in `utils/config.py`)

---

## 🛠 Tech Stack

### Programming & Frameworks
- **Python 3.9+**
- **Flask** – Backend Web Framework
- **HTML / CSS / JavaScript** – Frontend

### Machine Learning & Audio
- **PyTorch** – Deep Learning Framework
- **Swin Transformer** – Core model
- **Librosa** – Audio processing
- **NumPy, SciPy** – Numerical operations

---

