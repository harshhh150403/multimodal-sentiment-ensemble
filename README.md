# Multimodal Sentiment Analysis with Stacking Ensemble

A deep learning project for binary sentiment analysis using multimodal inputs (video + text) with an ensemble of three models.

## Overview

This project implements a **stacking ensemble** that combines three multimodal fusion models:

| Model | Text Encoder | Video Encoder | Fusion Method |
|-------|--------------|---------------|---------------|
| **IFFSA** | BERT | ResNet50 | Multi-Head Attention |
| **BFSA** | GPT-2 | VGG16 | Bilinear Fusion |
| **TBJE** | GloVe/BiLSTM | CNN+LSTM | Transformer |

A **Logistic Regression meta-learner** combines base model predictions for final classification.

## Project Structure

```
├── config.py          # Paths, hyperparameters, GPU setup
├── video_utils.py     # Video frame extraction & visualization
├── text_utils.py      # Tokenizer management & text preprocessing
├── data.py            # Dataset loading & caching
├── models.py          # IFFSA, BFSA, TBJE model definitions
├── ensemble.py        # Stacking ensemble logic
├── train.py           # Training & evaluation pipeline
├── main.py            # Entry point
├── requirements.txt   # Python dependencies
```

## Installation

1. **Clone or download** this project

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure dataset path** in `config.py`:
   ```python
   ROOT_FOLDER = "path/to/your/dataset"
   ```

## Dataset

This project uses the **CMU-MOSI** dataset structure:
```
root_folder/
├── Video/
│   └── Full/
│       ├── video1.mp4
│       ├── video2.mp4
│       └── ...
└── Transcript/
    └── Full/
        ├── video1.txt
        ├── video2.txt
        └── ...
```

## Usage

### Training

Run the complete training pipeline:
```bash
python main.py
```

This will:
1. Load and preprocess the dataset (with caching)
2. Train TBJE, IFFSA, and BFSA models
3. Train the stacking ensemble meta-learner
4. Evaluate all models and print results

### Output

Models are saved to `{ROOT_FOLDER}/saved_models/`:
- `iffsa_model/` - IFFSA Keras model
- `bfsa_model/` - BFSA Keras model
- `tbje_model/` - TBJE Keras model
- `ensemble_model.joblib` - Meta-learner (sklearn)
- `glove_tokenizer.pkl` - Fitted Keras tokenizer

## Key Features

- **Multimodal Fusion**: Combines text and video for richer sentiment understanding
- **Ensemble Learning**: Stacking ensemble improves over individual models
- **Data Caching**: Preprocessed video frames cached to disk for fast iteration
- **Memory Optimization**: GPU memory growth, session clearing between models
- **Reproducibility**: Random seeds set for consistent results

## Hardware Requirements

- **GPU**: Recommended (NVIDIA with 8GB+ VRAM)
- **RAM**: 16GB+ recommended
- **Storage**: ~10GB for cached data

## Configuration

Key hyperparameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_FRAMES` | 100 | Frames sampled per video |
| `MAX_TEXT_LENGTH` | 128 | Max tokens per text |
| `BATCH_SIZE` | 16 | Training batch size |
| `EPOCHS` | 1 | Training epochs (increase for better results) |
| `LEARNING_RATE` | 2e-5 | Adam learning rate |

## Documentation

See **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)** for:
- Detailed architecture explanations
- Model deep dives with diagrams
- Data pipeline documentation
- Interview Q&A
- Challenges faced and solutions

## Technologies Used

- **TensorFlow/Keras** - Deep learning framework
- **HuggingFace Transformers** - BERT, GPT-2
- **OpenCV** - Video processing
- **scikit-learn** - Ensemble meta-learner
- **VADER** - Sentiment label extraction




