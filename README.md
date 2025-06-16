# 🎭 Deepfake Detection using Video Frame Analysis (WIP)

This repository contains a work-in-progress project aimed at detecting deepfake videos by analyzing sequences of extracted  features using a combination of CNNs (ResNet50) and RNNs (LSTM, GRU) with attention mechanisms. The goal is to leverage temporal patterns across frames to distinguish between real and fake videos.

---

## 📁 Project Structure

📂 NEW DEEPFAKE
│
├── dataset/ # New dataset (face regions from Celeb-DF)
├── raw_videos/ # Original video sources (if needed)
├── all_features.pt # ResNet50-extracted features from video frames
├── all_labels.pt # Corresponding binary labels (0: real, 1: fake)
├── *.pth # Saved trained model weights
│
├── extract_frames.ipynb # Frame extraction logic using ffmpeg
├── FeatureExtraction.ipynb # Feature extractor using ResNet50
├── Resnet50Training.ipynb # Optional: Train/finetune ResNet50 (if needed)
├── LSTM_Deepfake_Training.ipynb
├── enhanced_lstm.ipynb
├── EmsembleLSTM.ipynb # Main ensemble pipeline (multiple LSTM/GRU models)


---

## 🧠 Model Architectures

All models are fed with pre-extracted frame-level features using **ResNet50**, reducing the computational overhead of per-frame CNN processing during training.

### 1. **LSTMClassifier**
- 2-layer Bi-directional LSTM
- Attention pooling over time dimension
- Followed by BatchNorm and FC layers
- Designed to capture temporal consistency and artifacts

### 2. **GRUClassifier**
- Same as above but uses GRUs instead of LSTMs
- Chosen for lower computational cost and faster convergence

### 3. **Ensemble of LSTM and GRU**
- Multiple models with varying hidden sizes and dropout settings:
    - `model1_lstm`: Bi-LSTM with hidden size 768
    - `model2_lstm`: Bi-LSTM with hidden size 512
    - `model3_gru`: Bi-GRU with hidden size 768
- Predictions are averaged via softmax probability for more robust inference

---

## 🔍 Why Multiple Models?

This project is still under heavy experimentation and development. Multiple model variants were trained to:
- Compare LSTM vs GRU performance
- Evaluate the impact of hidden dimensions and dropout
- Create an ensemble that improves generalization

The ensemble approach was found to outperform individual models on the test set.

---

## 🧪 Dataset Info

- Current dataset: **Facial regions from Celeb-DF**
- Every 60th frame from deepfakes and every 30th from real videos
- Already split into `train`, `val`, and `test` (70:20:10)

---

## 🚧 Work in Progress

This repository is still under active development. Upcoming improvements:
- Incorporation of frame-level augmentations
- Better handling of short vs long video clips
- Transition to lightweight models for deployment
- Frame importance visualization (attention heatmaps)

---

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install torch torchvision matplotlib
2. Run the pipeline in order:
extract_frames.ipynb

FeatureExtraction.ipynb

EmsembleLSTM.ipynb

📌 Author
Sainava Modak

Feel free to open issues or suggest improvements!