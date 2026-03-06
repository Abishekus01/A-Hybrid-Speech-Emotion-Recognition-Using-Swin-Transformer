# Hybrid Speech Emotion Recognition using Swin Transformer

## Overview

This project implements a Speech Emotion Recognition (SER) system using a hybrid deep learning architecture based on the Swin Transformer. The system analyzes speech signals and classifies them into different emotional categories such as happy, sad, angry, and neutral.

Speech Emotion Recognition is an important task in the field of human-computer interaction. By identifying emotions from speech, machines can respond more naturally and intelligently in applications such as virtual assistants, healthcare monitoring, customer service systems, and interactive learning platforms.

The proposed system combines traditional acoustic feature extraction with deep learning models to achieve improved performance in emotion classification.

---

## Project Architecture

The system follows a pipeline consisting of the following stages:

1. Audio Input  
2. Preprocessing  
3. Feature Extraction  
4. Spectrogram Generation  
5. Feature Fusion  
6. Swin Transformer Model  
7. Emotion Classification

---

## Feature Extraction

To represent the speech signal effectively, several acoustic features are extracted from the audio data.

### Mel Spectrogram

A Mel Spectrogram is a visual representation of the frequency spectrum of a signal over time.  
It converts the standard frequency scale into the Mel scale, which better represents how humans perceive sound.

The Mel Spectrogram helps deep learning models identify patterns related to speech characteristics and emotions.

---

### Zero Crossing Rate (ZCR)

Zero Crossing Rate measures how many times the audio waveform crosses the zero amplitude axis.

It is useful for identifying:
- Noisy signals
- Unvoiced speech sounds
- Signal intensity changes

Higher ZCR values often correspond to energetic or noisy speech.

---

### Chroma Features

Chroma features represent the distribution of energy across the 12 pitch classes of the musical octave.

In speech analysis, chroma helps identify tonal characteristics and harmonic structures present in the audio signal.

These features are useful for capturing pitch-related emotional cues.

---

### Root Mean Square Energy (RMSE)

Root Mean Square Energy measures the loudness or intensity of the audio signal.

It reflects how strong or weak the speech signal is at different time intervals. Emotional speech often shows noticeable variations in energy levels.

---

### Spectral Centroid

Spectral Centroid represents the center of mass of the spectrum.

It indicates where the "brightness" of the sound is located in the frequency spectrum. Higher values typically indicate sharper or brighter sounds.

---

### Spectral Bandwidth

Spectral Bandwidth measures the spread of frequencies around the spectral centroid.

It helps determine whether the sound contains concentrated or widely distributed frequency components.

---

### Spectral Contrast

Spectral Contrast measures the difference between peaks and valleys in the frequency spectrum.

This feature captures variations in harmonic structure and can help distinguish different emotional tones in speech.

---

## Model Architecture

The model uses a Swin Transformer architecture for classification.

The architecture includes the following components:

### Patch Tokenization

The input spectrogram image is divided into small patches. Each patch is converted into a token that can be processed by the transformer model.

---

### Swin Transformer Blocks

Swin Transformer blocks apply self-attention within local windows of the image. This approach improves computational efficiency while still capturing important contextual information.

The main components include:

Layer Normalization  
Window-based Multi-Head Self Attention (W-MSA)  
Shifted Window Multi-Head Self Attention (SW-MSA)  
Multi-Layer Perceptron (MLP)

---

### Feature Flattening

After transformer processing, the feature maps are flattened into a vector representation suitable for classification.

---

### Linear Classification Layer

A fully connected linear layer maps the extracted features to the number of emotion classes.

---

### Softmax Layer

The Softmax function converts the output values into probabilities for each emotion class.  
The class with the highest probability is selected as the predicted emotion.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Abishekus01/A-Hybrid-Speech-Emotion-Recognition-Using-Swin-Transformer.git
```

Move into the project directory:

```bash
cd A-Hybrid-Speech-Emotion-Recognition-Using-Swin-Transformer
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the training or inference script using:

```bash
python main.py
```

The system will load the trained model, process the input speech data, and predict the corresponding emotion.

---

## Evaluation Metrics

The model performance is evaluated using several metrics:

Accuracy  
Training Loss  
Validation Loss  
Validation Accuracy  
Precision  
Recall  
F1 Score  
Confusion Matrix

These metrics provide a comprehensive evaluation of the model’s classification performance.

---

## Applications

Speech Emotion Recognition can be used in many real-world applications:

Human Computer Interaction  
Mental Health Monitoring  
Customer Service Analytics  
Virtual Assistants  
Call Center Emotion Detection  
E-learning Systems

---

## Contributing

Contributions are welcome. If you would like to improve this project, please create a pull request or open an issue describing your proposed changes.

---

## License

This project is licensed under the MIT License. Refer to the LICENSE file for more details.

---

## Author

Abishek U S  
B.Tech Computer Science and Engineering  
SASTRA Deemed University

---

## Acknowledgments

This project builds upon research in the fields of speech processing, acoustic feature extraction, and transformer-based deep learning architectures. We acknowledge the contributions of the open-source community and researchers whose work has supported advancements in speech emotion recognition.
