# 🧠 BERT Question Answering (SQuAD) — Work in Progress 🚧

## 📌 Overview

This project explores **Question Answering (Q&A)** using **BERT** fine-tuned on the **SQuAD dataset**.

The goal is to build a model that can extract precise answers from a given context paragraph.

⚠️ **Note:** This project is currently a work in progress. The model is not yet producing reliable results, and improvements are ongoing.

---

## 🚀 Features

* BERT-based extractive Question Answering
* Training pipeline using SQuAD dataset
* Inference script for predictions
* Experimental implementation of Attention mechanisms
* Continuous development with daily commits

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

## 🧪 Example

**Context:**

```id="s1d4zc"
BERT is a transformer-based model developed by Google for natural language processing tasks.
```

**Question:**

```id="jx3k1t"
What type of model is BERT?
```

**Predicted Answer (current):**

```id="rf8u2p"
[Unreliable / still improving]
```


---

## 🧪 Attention Experiment

An additional experiment was conducted to better understand **attention mechanisms**.

* Custom attention implementation
* Helps visualize how the model focuses on relevant parts of the input
* Code available in `src/attention.py`

---

## 📈 Current Challenges

* Incorrect answer span predictions
* Training instability
* Data preprocessing issues (under investigation)

---

## 📈 Next Steps

* Fix training pipeline
* Debug tokenization and answer alignment
* Evaluate properly using EM and F1
* Improve model performance
* Compare with baseline results

---

## 🧠 Learning Goals

* Understand BERT internals
* Master extractive Question Answering
* Explore attention mechanisms in depth

---

## 🤝 Contributing

Suggestions and feedback are welcome.

---

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.
