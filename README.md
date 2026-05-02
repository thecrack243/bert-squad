<p align="center">
  <img src="img/bert_model.png" alt="Bert Model Banner"/>
</p>

# 🧠 BERT Question Answering (SQuAD)

## 📌 Overview

This project implements a **Question Answering (Q&A)** system using **BERT (Bidirectional Encoder Representations from Transformers)**.
The model is fine-tuned on the **SQuAD (Stanford Question Answering Dataset)** to extract answers from a given context paragraph.

---

## 🚀 Features

* Fine-tuned BERT model for extractive Q&A
* Uses SQuAD dataset for training and evaluation
* Predicts answer spans from context
* Clean training and inference pipeline
* Modular and reusable code

---

## 🗂️ Project Structure

```id="u3hs1m"
bert-squad/
│
├── img/
│
├── notebooks/
│   ├── attention_experiment.ipynb
│   ├── bert_experiment.ipynb
├── src/
│   ├── attentio.py
│   ├── bert.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

```bash id="b0v3s0"
git clone https://github.com/thecrack243/bert-squad.git
cd bert-squad
pip install -r requirements.txt
```

---

## ▶️ Usage

### Train the model

```bash id="bl3e2z"
python src/bert.py
```

---

---

## 🧪 Example

**Context:**

```id="d8fpzw"
Tesla was founded in 2003 and is headquartered in Austin.
```

**Question:**

```id="ojnddh"
Where is Tesla headquartered?
```

**Predicted Answer:**

```id="n9y5t3"
austin
```

---

## 📊 Model Details

* Model: `bert-base-uncased`
* Task: Extractive Question Answering
* Dataset: SQuAD
* Framework: PyTorch
* Library: Hugging Face Transformers

---

## 📈 Future Improvements

* Support for SQuAD v2 (unanswerable questions)
* Add evaluation metrics (Exact Match, F1 score)
* Deploy as an API (FastAPI / Flask)
* Add web interface for interactive Q&A

---

## 🤝 Contributing

Feel free to contribute or suggest improvements.

---

## 📄 License

MIT License
