# Deepfake Detection 

This repository implements a **RGB + Optical Flow + LSTM pipeline** for deepfake video detection. It uses frame-level ResNet18 features fused with optical flow and a temporal LSTM for full sequence prediction.

---

## 📂 Project Structure

```
NEW_DEEPFAKE/
├── dataset/
│   ├── rgb/                  # RGB frames extracted from raw videos
│   ├── optical_flow/         # Computed optical flow images
│   ├── inference_rgb/        # RGB frames for inference
│   ├── inference_flow/       # Optical flow for inference
├── features/                 # Saved (T, 1024) fused features
├── inference/
│   ├── videos/               # Videos for final prediction (real/fake)
│   ├── inference_notebook.ipynb # Notebook to run inference on new videos
├── models/                   # Saved best models (MixedResNet + LSTM)
├── raw_videos/               # Original raw videos
├── src/
│   ├── extract_rgb_frames.py # Extract RGB frames
│   ├── extract_optical_flow.py # Compute optical flow frames
│   ├── match_rgb_OpticalFlow.py # Verify RGB-Flow match
│   ├── extract_features.py   # Extract fused features
│   ├── train_mixed_model.py  # Train fused RGB+Flow ResNet
│   ├── train_LSTM.py         # Train LSTM on sequences
│   ├── train_rgb_cnn.py      # (Optional RGB only)
│   ├── train_of_cnn.py       # (Optional Optical Flow only)
├── utils/                    # Helper utilities
├── README.md
```

---

## ✅ What Each Script Does

| File                                 | Purpose                                                     |
| ------------------------------------ | ----------------------------------------------------------- |
| `extract_rgb_frames.py`              | Extract 32 RGB frames per video into `dataset/rgb/`         |
| `extract_optical_flow.py`            | Compute optical flow between consecutive frames             |
| `match_rgb_OpticalFlow.py`           | Check for any mismatches                                    |
| `extract_features.py`                | Run frozen MixedResNet on RGB+Flow → save (T,1024) features |
| `train_mixed_model.py`               | Train fused RGB+Flow ResNet classifier                      |
| `train_LSTM.py`                      | Train video-level Bi-LSTM on features                       |
| `inference/inference_notebook.ipynb` | End-to-end prediction on new videos                         |

---

## 🚦 Pipeline Order

1️⃣ **Extract RGB Frames:**

```bash
python src/extract_rgb_frames.py
```

2️⃣ **Compute Optical Flow:**

```bash
python src/extract_optical_flow.py
```

3️⃣ **Check Matching:**

```bash
python src/match_rgb_OpticalFlow.py
```

4️⃣ **Train Mixed ResNet:**

```bash
python src/train_mixed_model.py
```

5️⃣ **Extract Features:**

```bash
python src/extract_features.py
```

6️⃣ **Train LSTM:**

```bash
python src/train_LSTM.py
```

7️⃣ **Run Final Inference:**

* Use `inference/inference_notebook.ipynb`
* Place new videos in `inference/videos/real` and `inference/videos/fake`

---

## 📌 Author

**Sainava Modak**


---

**Performance:** 97% test accuracy on seen FF++ data, \~80% on unseen CelebDF samples.
**Architecture:** ResNet18 RGB + Optical Flow + Bi-LSTM.

