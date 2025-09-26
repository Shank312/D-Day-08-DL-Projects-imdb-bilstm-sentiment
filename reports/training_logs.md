

# Training Logs – IMDB Sentiment BiLSTM

## Experiment Setup
- **Dataset**: IMDB (25k train / 25k test, from `tensorflow_datasets`)
- **Validation Split**: 10% of training (2.5k samples)
- **Tokenizer/Vectorizer**: Keras `TextVectorization`
  - max_tokens = 20,000
  - sequence_length = 256
- **Model**:
  - Embedding(128, mask_zero=True)
  - SpatialDropout1D(0.2)
  - BiLSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
  - BiLSTM(32, dropout=0.2, recurrent_dropout=0.2)
  - Dropout(0.3)
  - Dense(1, activation="sigmoid")
- **Optimizer**: Adam (lr=1e-3)
- **Loss**: Binary Crossentropy
- **Metrics**: Accuracy, ROC-AUC
- **Callbacks**: EarlyStopping (patience=2, monitor=val_roc_auc), ModelCheckpoint, ReduceLROnPlateau

---

## Training Metrics
- **Epoch 1**: val_acc ≈ 0.846, val_roc_auc ≈ 0.938
- **Epoch 2**: val_acc ≈ 0.850, val_roc_auc ≈ 0.938
- **Epoch 3**: val_acc ≈ 0.857, val_roc_auc ≈ 0.932
- (Early stopped at epoch 4)

---

## Test Results
- **ROC-AUC**: ~0.905
- **Accuracy**: ~0.819
- **Precision/Recall/F1**:


precision recall f1-score support
0 0.7895 0.8703 0.8279 12500
1 0.8555 0.7679 0.8094 12500
accuracy 0.8191 25000

- **Confusion Matrix**:
[[10879 1621]
[ 2901 9599]]



---

## Error Analysis
- **False Negatives (predicted 0 but true 1)**: ~2900
- **False Positives (predicted 1 but true 0)**: ~1600
- Most misclassifications occur on:
- Long reviews (>300 tokens).
- Reviews with sarcasm or mixed sentiment.

---

## Decision Threshold
- Optimized threshold (by F1-score): ~0.20
- Saved in `models/decision_threshold.npy`.

---

## Final Artifacts
- `models/bilstm_imdb_v1.keras` (full pipeline with TextVectorization)
- `models/decision_threshold.npy` (decision threshold)
- `src/inference.py` (inference wrapper)
- `src/app.py` (FastAPI service)
