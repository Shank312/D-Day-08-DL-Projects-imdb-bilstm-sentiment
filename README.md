# 🎬 IMDb Sentiment Analysis with BiLSTM

[![Made with TensorFlow](https://img.shields.io/badge/Made%20with-TensorFlow-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![GitHub stars](https://img.shields.io/github/stars/Shank312/D-Day-08-DL-Projects-imdb-bilstm-sentiment?style=social)](https://github.com/Shank312/D-Day-08-DL-Projects-imdb-bilstm-sentiment)

This project implements a **Bidirectional LSTM (BiLSTM)** model for sentiment classification on the **IMDb movie reviews dataset**.  
The pipeline is **production-ready**, with training logs, saved artifacts, and inference scripts.

---

## 📂 Project Structure
imdb-bilstm-sentiment/
│── data/
│ ├── imdb_csv/ # train/test CSVs
│── models/
│ ├── bilstm_imdb_v1.keras # final trained model
│ ├── decision_threshold.npy # chosen threshold
│── reports/
│ ├── training_logs.md # experiment details & results
│── src/
│ ├── app.py # FastAPI service
│ ├── inference.py # Inference wrapper
│── README.md


---

## 🧠 Model Architecture
- **TextVectorization**: `max_tokens=20000`, `sequence_length=256`
- **Embedding(128, mask_zero=True)**
- **SpatialDropout1D(0.2)**
- **BiLSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)**
- **BiLSTM(32, dropout=0.2, recurrent_dropout=0.2)**
- **Dropout(0.3)**
- **Dense(1, activation="sigmoid")**

Optimizer: **Adam (lr=1e-3)**  
Loss: **Binary Crossentropy**  
Metrics: **Accuracy, ROC-AUC**

---

## 📊 Results

| Metric         | Validation | Test   |
|----------------|------------|--------|
| Accuracy       | ~0.85      | 0.819  |
| ROC-AUC        | ~0.94      | 0.905  |
| Precision (pos)| —          | 0.856  |
| Recall (pos)   | —          | 0.768  |
| F1-score (pos) | —          | 0.809  |

Confusion Matrix (Test set):

[[10879 1621]
[ 2901 9599]]


---

## 🚀 Usage

### 1. Clone repo
```bash
git clone https://github.com/Shank312/D-Day-08-DL-Projects-imdb-bilstm-sentiment.git
cd D-Day-08-DL-Projects-imdb-bilstm-sentiment

2. Install dependencies:
pip install -r requirements.txt

3. Run inference
Python API
from src.inference import load_artifacts, predict

load_artifacts()  # load model + threshold

texts = [
    "Absolutely wonderful and beautifully acted.",
    "What a waste of time. Terrible..."
]

labels, probs = predict(texts)
print(labels, probs)

CLI
python src/inference.py --text "The movie was fantastic!"

4. Run FastAPI service
uvicorn src.app:app --reload

Then:
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "A masterpiece of cinema!"}'


📑 Reports

Detailed training setup, metrics, and error analysis are available in:

reports/training_logs.md


🏆 Highlights

End-to-end NLP pipeline (data → training → evaluation → deployment).

Production-ready inference via both CLI and FastAPI.

Saved threshold optimization (F1 / Youden J).

Clean GitHub structure for easy re-use.


📌 Next Steps

Upload to Hugging Face 🤗 Hub with Gradio demo.

Experiment with attention layer or transformer-based encoder for improved performance.


📜 License

This project is released under the MIT License.
