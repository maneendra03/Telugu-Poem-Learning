# CNN-Based Telugu Poem Learning & Interpretation System

> *A deep learning system using Convolutional Neural Networks to classify Telugu poems by poetic meter (Chandas), source (Satakam), and provide interpretation support.*

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Telugu Poem Text Input                        │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  DATA PREPROCESSING                                             │
│  • Unicode NFC normalization    • Remove _x000D_ tokens         │
│  • Telugu character filtering   • Length thresholding            │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  FEATURE ENGINEERING                                            │
│  • Keras Tokenizer (vocab=30,000)                               │
│  • Text → Integer sequences → Pad to 400 tokens                 │
│  • One-hot encode labels (chandas / class / source)             │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  CNN MODEL (H200 GPU Optimized)                                 │
│                                                                  │
│  Embedding(30000, 200)                                          │
│       ▼                                                          │
│  Conv1D(256, k=5) → BatchNorm → MaxPool(3)   [syllable level]  │
│       ▼                                                          │
│  Conv1D(128, k=3) → BatchNorm → MaxPool(3)   [phrase level]    │
│       ▼                                                          │
│  Conv1D(64, k=3) → BatchNorm → GlobalMaxPool [structure level] │
│       ▼                                                          │
│  Dropout(0.4) → Dense(256) → Dense(128)                        │
│       ▼                          ▼                               │
│  ┌─────────────┐    ┌──────────────────┐                        │
│  │ Chandas (8)  │    │ Source (28+)     │  ← Multi-task heads   │
│  │  Softmax     │    │  Softmax         │                        │
│  └─────────────┘    └──────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

## 🧠 Human Learning Inspiration

This system mirrors how humans learn to recognize poetic meter:

| Human Process | CNN Equivalent |
|---|---|
| Hearing individual syllables (laghu/guru) | **Embedding Layer** — learns syllable representations |
| Recognizing local rhythmic patterns | **Conv1D (kernel=5)** — detects 5-gram patterns like gaṇas |
| Identifying phrase-level meter structure | **Conv1D (kernel=3)** — captures broader rhythmic phrases |
| Grasping overall poem structure | **GlobalMaxPooling** — extracts dominant rhythmic features |
| Categorizing into known meters | **Dense + Softmax** — classifies into chandas types |

## 📁 Project Structure

```
CNN Telugu/
├── config.py                # Hyperparameters (H200-optimized)
├── data_preprocessing.py    # Load, clean, merge datasets
├── feature_engineering.py   # Tokenize, pad, encode labels
├── model.py                 # CNN architectures (single + multi-task)
├── train.py                 # Training pipeline with callbacks
├── evaluate.py              # Metrics, confusion matrix, curves
├── interpretation.py        # Meaning extraction + TF-IDF keywords
├── app.py                   # Streamlit web interface
├── main.py                  # CLI entry point
├── requirements.txt         # Python dependencies
├── Dataset/
│   ├── Chandassu_Dataset.csv
│   └── processed/
│       ├── telugu_poems.json
│       ├── telugu_train.json
│       ├── telugu_val.json
│       ├── telugu_test.json
│       └── telugu_stats.json
├── models/                  # Saved models & encoders
└── outputs/                 # Evaluation plots & reports
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models
```bash
# Train both single-task and multi-task CNN
python main.py --mode train

# Train only single-task (chandas prediction)
python main.py --mode single

# Train only multi-task (chandas + source)
python main.py --mode multi
```

### 3. Evaluate
```bash
python main.py --mode evaluate
```

### 4. Interactive Prediction
```bash
python main.py --mode predict
```

### 5. Web Interface
```bash
streamlit run app.py
```

## ⚙️ H200 GPU Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Batch Size | 128 | Large batches for H200 throughput |
| Embedding Dim | 200 | Rich representations with ample VRAM |
| Vocab Size | 30,000 | Larger vocabulary for Telugu |
| Conv Filters | 256/128/64 | 3-layer deep feature extraction |
| Mixed Precision | FP16 | 2× speedup on H200 Tensor Cores |
| Max Epochs | 30 | More training with early stopping |

## 📊 Dataset Summary

- **10,605 total poems** from 28+ satakams
- **4,643 with chandas labels** (8 meter types, 3 classes)
- **Split**: Train 80% / Val 10% / Test 10%

| Meter Type | Telugu | Class |
|---|---|---|
| aataveladi | ఆటవెలది | vupajaathi |
| kandamu | కందము | jaathi |
| teytageethi | తేటగీతి | vupajaathi |
| seesamu | సీసము | vupajaathi |
| mattebhamu | మత్తేభము | vruttamu |
| champakamaala | చంపకమాల | vruttamu |
| vutpalamaala | ఉత్పలమాల | vruttamu |
| saardulamu | శార్దూలము | vruttamu |

## 🔬 Research Extensions

- Replace embeddings with [FastText Telugu](https://fasttext.cc/docs/en/crawl-vectors.html)
- Add Bidirectional LSTM baseline for comparison
- Integrate Attention mechanism on top of CNN features
- Analyze misclassifications by meter type
- Ablation study on augmented data impact
- Study effect of poem length on classification accuracy

## 📄 Technical Stack

Python | TensorFlow/Keras | NumPy | Pandas | scikit-learn | Matplotlib | Seaborn | Streamlit
