# Hate Speech Detector

A deep learning text classifier that detects hate speech in social media content using fine-tuned RoBERTa.

Built to solve a specific problem: standard classifiers trained on imbalanced hate speech datasets score 95%+ accuracy by predicting "Safe" for everything — catching zero actual hate speech. This project fixes that.

## How It Works

- **Model**: RoBERTa (pre-trained on social media data, not formal text — domain match matters)
- **Head swap**: Replaced RoBERTa's default pre-training head with a custom Linear Classification Layer connected to the `[CLS]` token, outputting Safe/Hate probability via Sigmoid
- **Preprocessing**: Normalizes misspelled slang + converts emojis to text tags (`🤬` → `[angry_face]`) so embeddings can read emotional signal
- **Class weighting**: Applied inside the PyTorch loss function — misclassifying hate speech costs the model far more than a false positive, forcing it to care about the minority class
- **Evaluation**: Precision, Recall, and F1-Score — accuracy is a vanity metric on imbalanced datasets

## Stack

Python · PyTorch · RoBERTa · HuggingFace Transformers · FastAPI · Davidson Hate Speech Dataset

## Running Locally
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --reload
```

## Write-up

Full breakdown of the build, the accuracy trap, and the fix:
[Building a Hate Speech Classifier: Why 99% AI Accuracy is a Complete Lie](https://medium.com/@moseleydev/building-a-hate-speech-classifier-why-99-ai-accuracy-is-a-complete-lie-64a6103b5c5f)
