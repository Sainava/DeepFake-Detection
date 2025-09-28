# 🎬 DeepFake Detection - Complete Workflow Diagram

## 📊 **TRAINING PIPELINE**

```mermaid
graph TD
    A[Raw Video Dataset] --> B[RGB Frame Extraction]
    A --> C[Fake/Real Labels]
    B --> D[32 RGB Frames per Video @ 224x224]
    D --> E[Optical Flow Computation]
    E --> F[TV-L1 Optical Flow Images]
    
    D --> G[RGB Dataset Structure]
    F --> H[Flow Dataset Structure]
    G --> I[dataset/rgb/real/]
    G --> J[dataset/rgb/fake/]
    H --> K[dataset/optical_flow/real/]
    H --> L[dataset/optical_flow/fake/]
    
    I --> M[RGB-Flow Matching Check]
    J --> M
    K --> M
    L --> M
    
    M --> N[Fused RGB+Flow Training]
    N --> O[MixedResNet Architecture]
    O --> P[RGB ResNet18 Branch]
    O --> Q[Flow ResNet18 Branch]
    
    P --> R[512D RGB Features]
    Q --> S[512D Flow Features]
    R --> T[Feature Concatenation]
    S --> T
    T --> U[1024D Fused Features]
    
    U --> V[Classification Head]
    V --> W[Binary Cross-Entropy Loss]
    W --> X[Model Optimization]
    X --> Y[Best Mixed Model Saved]
    
    Y --> Z[Feature Extraction Phase]
    Z --> AA[Frozen Feature Extractor]
    AA --> BB[Per-Video Feature Sequences]
    BB --> CC[features/mixed/*.npy]
    
    CC --> DD[LSTM Training Phase]
    DD --> EE[Bi-LSTM Architecture]
    EE --> FF[Video-Level Classification]
    FF --> GG[Temporal Sequence Modeling]
    GG --> HH[Best LSTM Model Saved]
    HH --> II[Training Metrics Logged]
```

## 🔍 **INFERENCE PIPELINE**

```mermaid
graph TD
    A[New Unseen Videos] --> B[Place in inference/videos/]
    B --> C[real/*.mp4]
    B --> D[fake/*.mp4]
    
    C --> E[Batch Video Processing]
    D --> E
    E --> F[Frame Extraction Loop]
    F --> G[RGB Frames per Video]
    G --> H[Optical Flow Computation]
    H --> I[Flow Images per Video]
    
    G --> J[Load Frozen MixedResNet]
    I --> J
    J --> K[RGB-Flow Feature Fusion]
    K --> L[1024D Feature Vectors]
    L --> M[Temporal Sequence Formation]
    M --> N[Save Video Features]
    
    N --> O[Load Trained LSTM]
    O --> P[Video-Level Prediction]
    P --> Q[Softmax Probabilities]
    Q --> R[Real vs Fake Classification]
    R --> S[Confidence Scores]
    S --> T[Batch Results Collection]
    T --> U[Confusion Matrix Generation]
    U --> V[Performance Metrics]
```

## 🧠 **DETAILED MODEL ARCHITECTURE FLOW**

```mermaid
graph TD
    A[Input: RGB + Optical Flow Frame Pairs] --> B[RGB Branch Input]
    A --> C[Flow Branch Input]
    
    B --> D[ResNet18 RGB Backbone]
    C --> E[ResNet18 Flow Backbone]
    
    D --> F[RGB Conv Layers]
    E --> G[Flow Conv Layers]
    
    F --> H[Conv2D + BatchNorm + ReLU + MaxPool]
    G --> I[Conv2D + BatchNorm + ReLU + MaxPool]
    
    H --> J[RGB Feature Maps]
    I --> K[Flow Feature Maps]
    
    J --> L[RGB Global Average Pool]
    K --> M[Flow Global Average Pool]
    
    L --> N[512D RGB Features]
    M --> O[512D Flow Features]
    
    N --> P[Feature Concatenation Layer]
    O --> P
    P --> Q[1024D Fused Representation]
    
    Q --> R[Dropout Layer 0.5]
    R --> S[Linear Classification Head]
    S --> T[2-Class Output: Real/Fake]
    
    %% LSTM Temporal Processing
    Q --> U[Feature Extraction Mode]
    U --> V[Per-Frame Features Saved]
    V --> W[Video Sequence Formation]
    W --> X[Temporal Feature Matrix T×1024]
    
    X --> Y[Bi-LSTM Layer 1 - 512 Hidden]
    Y --> Z[Bi-LSTM Layer 2 - 512 Hidden]
    Z --> AA[Hidden State Concatenation]
    AA --> BB[Dropout 0.5]
    BB --> CC[Final Linear Layer]
    CC --> DD[Video-Level Prediction]
```

## 🔧 **DATA PREPROCESSING WORKFLOW**

```mermaid
graph TD
    A[Raw Videos in raw_videos/] --> B[real/ and fake/ folders]
    B --> C[extract_rgb_frames.py]
    C --> D[Frame Sampling Strategy]
    D --> E[Uniform 32 frames per video]
    E --> F[Resize to 224×224]
    F --> G[Save as JPG sequences]
    
    G --> H[extract_optical_flow.py]
    H --> I[Consecutive Frame Pairs]
    I --> J[TV-L1 Optical Flow Algorithm]
    J --> K[Flow Normalization & Clipping]
    K --> L[3-Channel Flow Images (x, y, magnitude)]
    L --> M[31 flow images per video]
    
    G --> N[match_rgb_OpticalFlow.py]
    M --> N
    N --> O[Frame-Flow Alignment Check]
    O --> P[Match Validation Report]
    P --> Q[Expected: 32 RGB → 31 Flow per video]
    
    Q --> R[Dataset Structure Ready]
    R --> S[Train/Val/Test Split]
    S --> T[Stratified Video-Level Split 70/20/10]
    T --> U[Frame-Level Data Loading]
```

## 📱 **INFERENCE NOTEBOOK WORKFLOW**

```mermaid
graph TD
    A[Jupyter Notebook Launch] --> B[Load Training Metrics JSON]
    B --> C[Plot Train/Val Accuracy Curves]
    C --> D[Import Model Classes & Utils]
    D --> E[Scan inference/videos/ directory]
    
    E --> F[Build Video List from real/ + fake/]
    F --> G[Create Output Directories]
    G --> H[Process Each Video in Batch]
    
    H --> I[Extract RGB Frames to temp]
    I --> J[Compute TV-L1 Optical Flow]
    J --> K[Load Pre-trained Model Weights]
    
    K --> L[MixedResNet Feature Extraction]
    L --> M[LSTM Temporal Classification]
    M --> N[Collect Predictions & Confidences]
    
    N --> O[Generate Confusion Matrix Heatmap]
    O --> P[Classification Report with F1-Score]
    P --> Q[Performance Visualization Plots]
    Q --> R[Results Analysis & Interpretation]
```

## 🎮 **COMPLETE TRAINING EXECUTION FLOW**

```mermaid
graph TD
    A[Setup Python Environment] --> B[Install Dependencies]
    B --> C[opencv-contrib-python, torch, torchvision, etc.]
    C --> D[Navigate to Project Root Directory]
    
    D --> E[Step 1: Extract RGB Frames]
    E --> F[python src/extract_rgb_frames.py]
    F --> G[Output: dataset/real/, dataset/fake/]
    
    G --> H[Step 2: Compute Optical Flow]
    H --> I[cd src && PYTHONPATH=.. python extract_optical_flow.py]
    I --> J[Output: dataset/optical_flow/real/, dataset/optical_flow/fake/]
    
    J --> K[Step 3: Verify Data Alignment]
    K --> L[python src/match_rgb_OpticalFlow.py]
    L --> M[Validation: 32 RGB → 31 Flow matches]
    
    M --> N[Step 4: Train Fused CNN]
    N --> O[python src/train_mixed_model.py]
    O --> P[Output: models/mixed_model/best_mixed_model.pth]
    
    P --> Q[Step 5: Extract Features]
    Q --> R[python src/extract_features.py]
    R --> S[Output: features/mixed/*.npy sequences]
    
    S --> T[Step 6: Train LSTM]
    T --> U[python src/train_LSTM.py]
    U --> V[Output: models/lstm_model/best_lstm.pth + metrics]
    
    V --> W[Step 7: Run Inference]
    W --> X[jupyter notebook inference/inference_notebook.ipynb]
```

## 🔄 **SYSTEM COMPONENT INTERACTION**

```mermaid
graph TD
    A[Raw Video Files .mp4] --> B[OpenCV Frame Extraction]
    A --> C[Ground Truth Labels from Folders]
    
    B --> D[CV2 Video Capture & Processing]
    D --> E[TV-L1 Optical Flow Utils Module]
    E --> F[Landmark Feature Preprocessing]
    
    F --> G[PyTorch DataLoaders & Transforms]
    G --> H[MixedResNet Training Loop]
    H --> I[Model Checkpointing & Early Stop]
    
    I --> J[Frozen Feature Extraction Pipeline]
    J --> K[Bi-LSTM Training Loop with Scheduler]
    K --> L[JSON Metrics Logging System]
    
    L --> M[Training Performance Data]
    M --> N[Interactive Inference Notebook]
    N --> O[Matplotlib + Seaborn Visualization]
    O --> P[Confusion Matrix & Results Analysis]
```

## 🔄 **COMPLETE END-TO-END FLOW**

```mermaid
graph TD
    A[DeepFake Video Dataset] --> B[RGB Frame Sampling Strategy]
    B --> C[TV-L1 Optical Flow Generation]
    C --> D[Multi-Modal CNN Architecture]
    
    D --> E[RGB ResNet18 Spatial Branch]
    D --> F[Flow ResNet18 Motion Branch]
    
    E --> G[RGB Spatial Feature Learning]
    F --> H[Flow Motion Feature Learning]
    G --> I[Feature Fusion Concatenation]
    H --> I
    I --> J[Frame-Level 1024D Features]
    J --> K[Temporal LSTM Video Modeling]
    K --> L[Video-Level Binary Classification]
    L --> M[Real vs Fake Decision Boundary]
    M --> N[Confidence Score Assessment]
    N --> O[Performance Evaluation Metrics]
```

---

## 🎯 **SIMPLIFIED PRESENTATION WORKFLOW** 
*Perfect for slides and presentations*

### **Training Pipeline - Horizontal Flow**

```mermaid
graph LR
    A[Raw Videos] --> B[RGB+Flow<br/>Extraction] --> C[MixedResNet<br/>CNN Training] --> D[Feature<br/>Extraction] --> E[Bi-LSTM<br/>Training] --> F[Trained<br/>Models]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style D fill:#fff3e0
    style E fill:#fce4ec
    style F fill:#e0f2f1
```

### **Real-Time Inference Pipeline - Horizontal Flow**

```mermaid
graph LR
    A[📹 New Videos] --> B[🔍 Frame<br/>Processing] --> C[🌊 Optical<br/>Flow] --> D[🧠 Feature<br/>Extraction] --> E[⏱️ Temporal<br/>LSTM] --> F[📊 Classification] --> G[✅ Real/Fake<br/>Result]
    
    style A fill:#e1f5fe
    style B fill:#f0f4c3
    style C fill:#e1f5fe
    style D fill:#e8f5e8
    style E fill:#fff3e0
    style F fill:#f3e5f5
    style G fill:#fce4ec
```

### **Model Architecture - Essential Flow**

```mermaid
graph LR
    A[RGB+Flow<br/>Input] --> B[Dual ResNet18<br/>Branches] --> C[Feature<br/>Concatenation] --> D[Bi-LSTM<br/>Temporal] --> E[Binary<br/>Classification]
    
    A -.-> A1[RGB Frames<br/>224x224x3]
    A -.-> A2[Flow Images<br/>224x224x3]
    B -.-> B1[Spatial Features<br/>512D each]
    C -.-> C1[Fused Features<br/>1024D]
    D -.-> D1[Sequence Modeling<br/>T×1024 → 1024]
    E -.-> E1[Real vs Fake<br/>Confidence]
    
    style A fill:#bbdefb
    style B fill:#c8e6c9
    style C fill:#dcedc8
    style D fill:#ffe0b2
    style E fill:#f8bbd9
```

### **Data Flow - Core Components**

```mermaid
graph LR
    A[Video<br/>Input] --> B[Frame<br/>Extraction] --> C[Flow<br/>Computation] --> D[CNN<br/>Processing] --> E[LSTM<br/>Analysis] --> F[DeepFake<br/>Detection]
    
    A -.-> A1[MP4 Files<br/>Various Lengths]
    B -.-> B1[32 Frames<br/>@ 224x224]
    C -.-> C1[TV-L1<br/>Optical Flow]
    D -.-> D1[RGB+Flow<br/>Fusion CNN]
    E -.-> E1[Temporal<br/>Pattern Analysis]
    F -.-> F1[Confidence<br/>Score Output]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style D fill:#fff3e0
    style E fill:#fce4ec
    style F fill:#e0f2f1
```

---

## 📋 **Key Workflow Stages Summary**

| Stage | Input | Process | Output | Duration |
|-------|-------|---------|--------|----------|
| **Data Preparation** | Raw videos | Frame extraction + Optical flow | RGB/Flow datasets | ~30 min |
| **CNN Training** | Frame pairs | ResNet18 dual-branch fusion | Frame-level classifier | ~2-4 hours |
| **Feature Extraction** | Trained CNN | Frozen feature extraction | Video sequences | ~15 min |
| **LSTM Training** | Feature sequences | Bi-LSTM temporal modeling | Video classifier | ~1-2 hours |
| **Inference** | New videos | End-to-end pipeline | Real/Fake predictions | ~5 min/video |
| **Evaluation** | Test results | Confusion matrix + metrics | Performance analysis | ~2 min |

---

## 🎯 **Critical Decision Points**

1. **Frame Sampling**: Fixed 32 frames → Uniform temporal coverage across video lengths
2. **Flow Algorithm**: TV-L1 → Accurate dense motion estimation between consecutive frames  
3. **CNN Architecture**: Dual ResNet18 → RGB spatial + Flow motion feature fusion
4. **Feature Fusion**: Concatenation → 1024D joint representation (512+512)
5. **Temporal Modeling**: Bi-LSTM → Forward + backward sequence classification
6. **Training Strategy**: CNN first, then LSTM → Staged learning for stability

## 📊 **DETAILED COMPONENT BREAKDOWN**

### **MixedResNet Architecture Details**
```python
# RGB Branch: ResNet18 → 512D features
# Flow Branch: ResNet18 → 512D features  
# Fusion: Concatenate(RGB, Flow) → 1024D
# Head: Dropout(0.5) → Linear(1024, 2) → Softmax
```

### **Bi-LSTM Configuration** 
```python
# Input: (Batch, Sequence_Length, 1024)
# LSTM: 2 layers, 512 hidden, bidirectional=True
# Output: Hidden states → Dropout(0.5) → Linear(1024, 2)
```

### **Training Hyperparameters**
| Parameter | MixedResNet | Bi-LSTM |
|-----------|-------------|---------|
| **Batch Size** | 8 | 16 |
| **Learning Rate** | 1e-4 | 1e-3 |
| **Epochs** | 12 | 30 |
| **Optimizer** | Adam | Adam |
| **Scheduler** | StepLR | ReduceLROnPlateau |
| **Early Stop** | 3 epochs | N/A |

---

## 📋 **Quick Reference - Key Technologies**

| Component | Purpose | Technology | Key Features |
|-----------|---------|------------|--------------|
| **OpenCV** | Video frame extraction | Computer Vision | TV-L1 optical flow, frame sampling |
| **TV-L1 Flow** | Motion detection between frames | Optical Flow Algorithm | Dense, accurate motion vectors |
| **ResNet18** | Spatial feature extraction | Convolutional Neural Network | Pre-trained, efficient, proven |
| **Bi-LSTM** | Temporal sequence modeling | Recurrent Neural Network | Forward+backward context |
| **PyTorch** | Deep learning framework | Machine Learning Platform | GPU acceleration, auto-grad |
| **Jupyter** | Interactive inference | Data Science Environment | Visualization, experimentation |

---

## 🏗️ **PROJECT STRUCTURE OVERVIEW**

```
DeepFake-Detection/
├── 📁 raw_videos/           # Input dataset
│   ├── real/*.mp4          # Authentic videos
│   └── fake/*.mp4          # DeepFake videos
├── 📁 dataset/             # Processed frames
│   ├── rgb/real/           # RGB frame sequences  
│   ├── rgb/fake/           # RGB frame sequences
│   ├── optical_flow/real/  # Flow image sequences
│   └── optical_flow/fake/  # Flow image sequences
├── 📁 src/                 # Training scripts
│   ├── extract_rgb_frames.py      # Step 1: Frame extraction
│   ├── extract_optical_flow.py    # Step 2: Flow computation
│   ├── match_rgb_OpticalFlow.py   # Step 3: Data validation
│   ├── train_mixed_model.py       # Step 4: CNN training
│   ├── extract_features.py        # Step 5: Feature extraction
│   └── train_LSTM.py              # Step 6: LSTM training
├── 📁 models/              # Saved model weights
│   ├── mixed_model/best_mixed_model.pth
│   └── lstm_model/best_lstm.pth
├── 📁 features/            # Extracted features
│   └── mixed/*.npy         # Video-level feature sequences
├── 📁 inference/           # Inference pipeline
│   ├── videos/real/        # Test videos (real)
│   ├── videos/fake/        # Test videos (fake)  
│   └── inference_notebook.ipynb   # Interactive inference
└── 📁 utils/               # Helper utilities
    └── optical_flow_utils.py      # Flow computation functions
```

---

## 🎯 **Performance Expectations**

### **Reported Accuracy**
- **Training Data (FF++)**: ~97% test accuracy
- **Unseen Data (CelebDF)**: ~80% cross-dataset accuracy
- **Architecture**: ResNet18 RGB + Optical Flow + Bi-LSTM

### **Computational Requirements** 
- **GPU Memory**: ~4-6GB VRAM for training
- **Training Time**: ~3-6 hours total pipeline
- **Inference Speed**: ~30 seconds per video (CPU)
- **Storage**: ~2-5GB for processed datasets

---

## 🎯 **One-Line Summary**
**Raw videos → RGB+Optical Flow extraction → Dual CNN spatial-temporal fusion → Bi-LSTM sequence modeling → DeepFake binary classification**

This comprehensive workflow represents the complete DeepFake Detection pipeline from raw video data to final real/fake classification with detailed technical specifications! 🚀

---

## 🔗 **Related Documentation**
- [Training Scripts Documentation](./src/)
- [Model Architecture Details](./models/)
- [Inference Examples](./inference/)
- [Utility Functions](./utils/)

---

*Last Updated: September 2025*
*Repository: DeepFake-Detection by Sainava*