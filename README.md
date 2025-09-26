# 🎬 IMDb Sentiment Analysis with BiLSTM

[![Made with TensorFlow](https://img.shields.io/badge/Made%20with-TensorFlow-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![GitHub stars](https://img.shields.io/github/stars/Shank312/D-Day-08-DL-Projects-imdb-bilstm-sentiment?style=social)](https://github.com/Shank312/D-Day-08-DL-Projects-imdb-bilstm-sentiment)

This project implements a **Bidirectional LSTM (BiLSTM)** model for binary sentiment classification on the **IMDb movie reviews dataset**.  
The solution is **end-to-end production ready** — including training, evaluation, saved artifacts, and inference scripts.

---

## 📂 Repository Structure
imdb-bilstm-sentiment/
│── data/
│ ├── imdb_csv/ # train/test CSVs
│── models/
│ ├── bilstm_imdb_v1.keras # final trained model
│ ├── decision_threshold.npy # optimized threshold
│ ├── tokenizer.pkl # tokenizer (legacy pipeline)
│ ├── preproc.json # preprocessing config
│── reports/
│ ├── training_logs.md # metrics & results
│── src/
│ ├── app.py # FastAPI inference service
│ ├── inference.py # prediction wrapper
│── README.md


---

## 🧠 Model Architecture
- **TextVectorization**: `max_tokens=20,000`, `sequence_length=256`
- **Embedding(128, mask_zero=True)**
- **SpatialDropout1D(0.2)**
- **BiLSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)**
- **BiLSTM(32, dropout=0.2, recurrent_dropout=0.2)**
- **Dropout(0.3)**
- **Dense(1, activation="sigmoid")**

**Optimizer:** Adam (lr=1e-3)  
**Loss:** Binary Crossentropy  
**Metrics:** Accuracy, ROC-AUC  

---

## 📊 Results

| Metric         | Validation | Test   |
|----------------|------------|--------|
| Accuracy       | ~0.85      | 0.819  |
| ROC-AUC        | ~0.94      | 0.905  |
| Precision (pos)| —          | 0.856  |
| Recall (pos)   | —          | 0.768  |
| F1-score (pos) | —          | 0.809  |

**Confusion Matrix (Test set):**
[[10879 1621]
[ 2901 9599]]


---

## 🚀 Usage

### 1. Clone repository
```bash
git clone https://github.com/Shank312/D-Day-08-DL-Projects-imdb-bilstm-sentiment.git
cd D-Day-08-DL-Projects-imdb-bilstm-sentiment

2. Install dependencies
pip install -r requirements.txt

3. Python Inference
from src.inference import load_artifacts, predict

# Load trained model & threshold
load_artifacts()

texts = [
    "Absolutely wonderful and beautifully acted.",
    "What a waste of time. Terrible..."
]

labels, probs = predict(texts)
print(labels, probs)

4. CLI Inference
python src/inference.py --text "The movie was fantastic!"

5. Run FastAPI Service
uvicorn src.app:app --reload

Then test with:
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "A masterpiece of cinema!"}'


📑 Reports

See detailed training process, validation curves, and test evaluation in:
➡️ reports/training_logs.md


🏆 Highlights

End-to-end NLP pipeline: data → preprocessing → training → evaluation → deployment.

Production-ready inference via Python API, CLI, and FastAPI.

Optimized decision threshold (decision_threshold.npy) for best F1.

Clean repo structure for easy reuse and extension.


📌 Next Steps

Publish model on Hugging Face Hub with Gradio demo.

Experiment with attention mechanisms or transformer-based encoders for improved performance.


📜 License

This project is released under the MIT License.
