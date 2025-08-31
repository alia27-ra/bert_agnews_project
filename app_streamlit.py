#!/usr/bin/env python
import json
import os
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="AG News Topic Classifier (BERT)", page_icon="📰", layout="centered")

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join("models", "bert-agnews"))

@st.cache_resource
def load_pipe():
    return pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR, return_all_scores=True)

st.title("📰 AG News Topic Classifier (BERT)")
st.caption("Fine-tuned on AG News using `bert-base-uncased`")

pipe = load_pipe()

headline = st.text_area("Enter a news headline", placeholder="e.g., Apple unveils new iPhone with upgraded camera", height=100)

col1, col2 = st.columns(2)

with col1:
    if st.button("Classify"):
        if not headline.strip():
            st.warning("Please enter a headline.")
        else:
            with st.spinner("Predicting..."):
                preds = pipe(headline)[0]  # list of dicts
                preds_sorted = sorted(preds, key=lambda x: x["score"], reverse=True)
                top = preds_sorted[0]
                st.success(f"**Predicted Topic:** {top['label']}  \n**Confidence:** {top['score']:.4f}")
                st.subheader("All class probabilities")
                st.table([{ "label": p["label"], "score": round(p["score"], 4)} for p in preds_sorted])

with col2:
    st.info("Labels were learned from AG News: World, Sports, Business, Sci/Tech.")
    st.code("streamlit run app_streamlit.py", language="bash")
