

import gradio as gr
from imdb_bilstm.pipeline import SentimentPipeline

pipe = SentimentPipeline("models", device="cpu")

def predict(text):
    out = pipe.predict_one(text)
    return f"{out['label'].upper()} ({out['prob']:.2f})"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Enter a review"),
    outputs=gr.Textbox(label="Prediction"),
    title="IMDB Sentiment (BiLSTM)",
    description="Trained BiLSTM sentiment classifier with ~86.5% accuracy."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)
