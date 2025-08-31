# AG News Topic Classifier (BERT)

This project fine-tunes `bert-base-uncased` on the AG News dataset to classify news headlines into:
- World
- Sports
- Business
- Sci/Tech

## 1) Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

> GPU is recommended for training. On CPU it can be slow.

## 2) Train

```bash
python train_bert_agnews.py
```

This will:
- Download AG News via Hugging Face Datasets
- Split a validation set (10%)
- Fine-tune BERT for 3 epochs
- Evaluate on validation + test (Accuracy, weighted F1)
- Save the best model to `models/bert-agnews`

## 3) Run Apps

### Streamlit
```bash
streamlit run app_streamlit.py
```

### Gradio
```bash
python app_gradio.py
```

Both apps will load the saved model from `models/bert-agnews` by default.
Set `MODEL_DIR` env var to override.

## 4) Project Structure

```
.
├── app_gradio.py
├── app_streamlit.py
├── models/
│   └── bert-agnews/          # created after training
├── requirements.txt
├── train_bert_agnews.py
└── README.md
```

## 5) Tips
- Adjust `MAX_LENGTH`, learning rate, or `num_train_epochs` inside `train_bert_agnews.py` for better results.
- Use a larger batch size if your GPU allows.
- For reproducibility, seed is fixed to 42.
