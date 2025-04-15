# 🧠 LipNet: End-to-End Sentence-Level Lip Reading Model

LipNet is a deep learning-based model for lip reading that takes video frames of a speaker and predicts the sentence being spoken using only visual information (no audio). This implementation is inspired by the original [LipNet paper](https://arxiv.org/abs/1611.01599) and uses a combination of 3D Convolutional Neural Networks, Bidirectional LSTMs, and CTC Loss.

## 📁 Project Structure

```
├── LipNet.ipynb        # Jupyter Notebook containing the complete pipeline
├── data/               # Folder for video samples and alignment data 
├── models/             # Trained model weights 
```

## 🚀 Features

- Video preprocessing with OpenCV
- Facial region extraction using MTCNN (or similar)
- Sequence modeling using 3D CNN and BiLSTM
- Connectionist Temporal Classification (CTC) loss for unaligned sequence prediction
- Inference pipeline for new video input

## 🛠️ Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- OpenCV
- `matplotlib`

## 📦 Dataset

This project uses the **GRID Corpus**, a large audio-visual sentence corpus. You must manually download and preprocess the dataset (or use a prepared version if available).

## 🧪 Training & Evaluation

The training pipeline involves:
- Loading and preprocessing the video frames
- Generating character-level labels from alignment files
- Training the LipNet model with CTC loss
- Evaluating using word error rate (WER) and character error rate (CER)


## 📚 References

- Assael et al., *LipNet: End-to-End Sentence-Level Lipreading*, [arXiv:1611.01599](https://arxiv.org/abs/1611.01599)
