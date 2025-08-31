#!/usr/bin/env python
import os
import gradio as gr
from transformers import pipeline

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join("models", "bert-agnews"))
pipe = pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR, return_all_scores=True)

def predict(text):
    if not text or not text.strip():
        return {"": 0.0}, {}
    scores = pipe(text)[0]
    scores_sorted = sorted(scores, key=lambda x: x["score"], reverse=True)
    label_scores = {s["label"]: float(s["score"]) for s in scores_sorted}
    top = max(label_scores, key=label_scores.get)
    return {top: label_scores[top]}, label_scores

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=3, placeholder="Type a news headline here...", label="Headline"),
    outputs=[gr.Label(num_top_classes=4, label="Top Prediction"), gr.JSON(label="All class probabilities")],
    title="AG News Topic Classifier (BERT)",
    description="Fine-tuned bert-base-uncased on AG News. Enter a headline to get the topic.",
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
