# ECG Arrhythmia Classification: Comparative Study of Neural Architectures

A comprehensive comparative analysis of four state-of-the-art neural network architectures for classifying cardiac arrhythmias using the MIT-BIH Arrhythmia Database.

## Project Overview

This project evaluates and compares the performance of different deep learning architectures for ECG beat classification using the standardized AAMI (Association for the Advancement of Medical Instrumentation) arrhythmia categories. The study investigates the impact of class balancing on model performance.

## Dataset

**Source:** MIT-BIH Arrhythmia Database (Kaggle)
- **Total Beats:** 110,861 ECG heartbeats
- **Sampling Rate:** 360 Hz
- **Signal Leads:** MLII (Primary) and V5 (Secondary)
- **Window Size:** 360 samples per beat (180 samples before and after the annotation point)

### AAMI Classification Categories

| Class | Name | Count (Imbalanced) | Percentage |
|-------|------|-------------------|-----------|
| 0 | Normal (N) | 90,589 | 81.7% |
| 1 | SVEB (S) - Supraventricular Ectopic Beat | 2,779 | 2.5% |
| 2 | VEB (V) - Ventricular Ectopic Beat | 7,236 | 6.5% |
| 3 | Fusion (F) | 803 | 0.7% |
| 4 | Unknown/Paced (Q) | 9,454 | 8.5% |

## Data Preprocessing

### Signal Processing Pipeline

1. **Bandpass Filtering:** Butterworth filter (0.5-50 Hz, order 4)
   - Removes low-frequency baseline wander and high-frequency noise
   
2. **Normalization:** Z-score standardization
   - Centers the signal around 0 with unit variance
   - Improves model convergence
   
3. **Beat Extraction:** 
   - Extracts 360-sample windows centered on annotation points
   - Excludes beats near signal boundaries (< 180 samples from start/end)
   
4. **Reshaping:** 
   - Original: (Samples, 360) → PyTorch format: (Samples, 1, 360)
   - For sequence models: (Samples, 360, 1)

### Train-Test Split
- Training: 80% (~88,688 beats)
- Testing: 20% (~22,173 beats)
- Random seed: 42 (reproducible split)

## Neural Network Architectures

### 1. CNN with Temporal Attention (CNN_Temporal_Attention)

**Purpose:** Extract local temporal patterns with learned attention weights

**Architecture:**
- Conv1D Layers: 2 blocks (32→64 channels)
- MaxPooling: Reduces sequence length (360→180→90)
- Temporal Attention Mechanism:
  - Learns importance scores for each timestep
  - Applies softmax to create attention weights (sum to 1.0)
  - Weighted sum creates context vector
- Classifier: Linear layers with ReLU and dropout (0.3)

**Key Innovation:** Attention mechanism identifies critical points in the heartbeat (e.g., QRS complex)

### 2. LSTM (Long Short-Term Memory)

**Purpose:** Capture long-range temporal dependencies in sequential data

**Architecture:**
- 2 LSTM layers (hidden size: 64)
- Bidirectional processing of sequences
- Dropout (0.2) between layers for regularization
- Extracts final hidden state for classification

**Advantage:** Handles variable-length sequences and learns temporal patterns

### 3. ResNet1D (Residual Network 1D)

**Purpose:** Enable training of very deep networks through skip connections

**Architecture:**
- Initial layer: Conv1d (kernel=7, stride=2) → MaxPool
- 2 Residual Blocks with skip connections
- Progressively reduces sequence length (360→180→90→45→23)
- Global Average Pooling flattens output
- Dense classifier

**Key Innovation:** Skip connections prevent vanishing gradients, enabling deeper architectures

### 4. Transformer

**Purpose:** Leverage self-attention for parallel processing of all timesteps

**Architecture:**
- Linear embedding: Projects 1 value → 64 dimensions
- Positional encoding: Learnable position embeddings
- Transformer encoder: 4 attention heads, 2 layers
- Self-attention computes relationships between all timesteps
- Global average pooling + classification head

**Advantage:** Captures long-range dependencies more efficiently than RNNs

## Training Configuration

### Hyperparameters
- **Optimizer:** Adam (learning rate: 0.001)
- **Loss Function:** CrossEntropyLoss
- **Batch Size:** 64
- **Epochs:** 10
- **Device:** GPU (CUDA if available)

### Class Weighting
For balanced training:
```
Weight = Total_Samples / (n_classes × samples_in_class)
```
Increases loss contribution for underrepresented classes

## Results

### Before Class Balancing

| Rank | Architecture | Peak Accuracy | Training Time |
|------|--------------|---------------|---------------|
| 1 | ResNet1D | **99.27%** | 1.6 min |
| 2 | Transformer | 98.51% | 8.9 min |
| 3 | LSTM | 96.75% | 1.4 min |
| 4 | CNN_Temporal_Attention | 96.01% | 1.4 min |

**Detailed Metrics (Best Model: ResNet1D)**
- Accuracy: 99.17%
- Precision: 99.16%
- Recall: 99.17%
- Macro F1-Score: 0.948
- Macro AUPRC: 0.962

### After Class Balancing

| Rank | Architecture | Peak Accuracy | Training Time |
|------|--------------|---------------|---------------|
| 1 | ResNet1D | **97.47%** | 1.6 min |
| 2 | Transformer | 94.46% | 8.9 min |
| 3 | CNN_Temporal_Attention | 89.55% | 1.1 min |
| 4 | LSTM | 89.01% | 1.4 min |

**Detailed Metrics (Best Model: ResNet1D)**
- Accuracy: 94.16%
- Precision: 97.37%
- Recall: 94.16%
- Macro F1-Score: 0.781
- Macro AUPRC: 0.963

## Key Findings

### 1. Imbalanced vs. Balanced Data Trade-offs

- **Imbalanced Training:** 
  - Higher overall accuracy (99.27%)
  - Biased toward majority class (Normal beats)
  - Poor minority class detection

- **Balanced Training:**
  - Moderate accuracy decrease (97.47%)
  - Better minority class recall
  - More clinically relevant (fewer missed arrhythmias)

### 2. Architecture Performance

- **ResNet1D Dominance:** Skip connections enable deeper learning
- **Transformer Trade-off:** Excellent performance but 5-6× slower training
- **LSTM Instability:** Showed higher variance in balanced setting
- **CNN+Attention:** Fast training with reasonable performance

### 3. Class Weighting Effect

- Reduces bias toward majority class
- Maintains high Macro AUPRC (0.963)
- Slightly decreases Macro F1 due to more balanced predictions
- Improves clinical utility despite lower overall accuracy

## Evaluation Metrics

### Accuracy
- Percentage of correct predictions
- Biased by class imbalance in imbalanced dataset

### Precision & Recall
- **Precision:** True Positives / (True Positives + False Positives)
- **Recall:** True Positives / (True Positives + False Negatives)

### Macro F1-Score
- Unweighted average of F1-scores across all classes
- Fair metric for imbalanced datasets

### Macro AUPRC (Area Under Precision-Recall Curve)
- Average of per-class Area Under Precision-Recall curves
- Better for imbalanced classification than ROC-AUC

## Visualization

### Shared Visualization Block

The project uses **the same visualization code block** for both imbalanced and balanced training results:

```python
# Identical visualization pipeline for both scenarios:
1. Confusion Matrices (4 subplots for each model)
2. Macro F1-Score Comparison Bar Chart
3. Macro AUPRC Comparison Bar Chart
4. ROC Curves (best performing model)
5. Precision-Recall Curves (best performing model)
```

**Why Reuse?**
- Consistent comparison framework
- Identical metrics for both scenarios
- Efficient code organization
- Demonstrates reproducibility

**Note:** The visualization code is called twice—once after imbalanced training and once after balanced training—to generate comparative insights.

### Output Visualizations

1. **Confusion Matrices:** Shows classification patterns per architecture
2. **ROC & PR Curves:** Per-class performance across decision thresholds
3. **Bar Charts:** Comparative model performance

## Hyperparameter Grid Search

### Parameters Tested

- **Learning Rates:** 0.001, 0.0001
- **Batch Sizes:** 32, 64
- **Balancing Modes:** Balanced, None
- **Architectures:** 4 models
- **Total Configurations:** 2 × 4 × 2 × 2 = 32

### Training Duration
- 5 epochs per configuration
- Imbalanced: ~2-3 hours total
- Balanced: ~2-3 hours total

## How to Run

### Prerequisites
```bash
pip install kaggle torch scikit-learn pandas numpy scipy matplotlib seaborn
```

### Steps

1. **Setup Kaggle Credentials**
   - Upload your `kaggle.json` file
   - Place in `/root/.kaggle/`

2. **Download Dataset**
   - Kaggle API automatically fetches MIT-BIH dataset
   - Unzips into `./mitbih_csv/`

3. **Run Notebook Cells in Order**
   1. Kaggle setup & download
   2. Dataset exploration
   3. Data loading & preprocessing
   4. Beat extraction & normalization
   5. Model architecture definitions
   6. Training (imbalanced)
   7. Visualization (imbalanced)
   8. Class weight calculation
   9. Training (balanced)
   10. Visualization (balanced)
   11. Hyperparameter grid search

## Project Structure

```
dl/
├── dl_minip (1).ipynb          # Main notebook with all code
├── results.txt                 # Training output (imbalanced & balanced)
├── README.md                   # This file
└── mitbih_csv/                 # Dataset (generated)
    ├── 100_ekg.csv
    ├── 100_annotations_1.csv
    ├── ... (47 more record files)
    └── annotation_symbols.csv
```

## References

- MIT-BIH Arrhythmia Database: https://www.physionet.org/content/mitdb/
- AAMI Standard: EC57 (Recommended Practice for Testing and Reporting Performance Results of Cardiac Rhythm and ST Segment Measurement Algorithms)
- ResNet: He, K., et al. (2015). Deep Residual Learning for Image Recognition.
- Transformer: Vaswani, A., et al. (2017). Attention Is All You Need.

## Author Notes

This comparative study demonstrates that:
- **ResNet1D is the optimal architecture** for ECG classification in both scenarios
- **Class balancing is essential** for clinical applications despite accuracy trade-off
- **Identical visualization blocks** provide consistent comparison frameworks
- **Deep learning models** can achieve >97% accuracy on AAMI classification

---

**DL Miniproject Members:**
- *Tanishque Suthar*
- *Aman Yadav*
- *Ashutoshkumar Tripathi*
- *Rajat Verma*
