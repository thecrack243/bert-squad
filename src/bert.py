# ============================================================
# BERT Question Answering Experiment on SQuAD
# Author: Emmanuel Ilunga
# Level: Undergraduate (Year 2)
# Purpose: Fine-tuning BERT for Question Answering
# ============================================================

# --------------------
# 1. Install libraries (only needed in Colab)
# --------------------
# Uncomment the next line if you run this in Colab
# !pip install -q transformers datasets torch

# --------------------
# 2. Imports
# --------------------
import torch
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForQuestionAnswering
from torch.optim import AdamW

# --------------------
# 3. Load SQuAD dataset
# --------------------
print("Loading SQuAD dataset...")
squad = load_dataset("squad")

example = squad["train"][0]
print("\nExample from dataset:")
print("Context:", example["context"])
print("Question:", example["question"])
print("Answer:", example["answers"]["text"][0])

# --------------------
# 4. Load tokenizer
# --------------------
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# --------------------
# 5. Feature preparation function
# Convert character-level answers to token-level positions
# --------------------
def prepare_features(example):
    encoding = tokenizer(
        example["question"],
        example["context"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    offset_mapping = encoding.pop("offset_mapping")[0]

    answer_start_char = example["answers"]["answer_start"][0]
    answer_text = example["answers"]["text"][0]
    answer_end_char = answer_start_char + len(answer_text)

    start_token = 0
    end_token = 0

    for idx, (start, end) in enumerate(offset_mapping):
        if start <= answer_start_char < end:
            start_token = idx
        if start < answer_end_char <= end:
            end_token = idx
            break

    encoding["start_positions"] = torch.tensor([start_token])
    encoding["end_positions"] = torch.tensor([end_token])

    return encoding

# --------------------
# 6. Prepare a small subset (demo purpose)
# --------------------
print("\nPreparing training features...")
samples = squad["train"].select(range(5))
features = [prepare_features(s) for s in samples]

# --------------------
# 7. Load BERT QA model
# --------------------
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
model.train()

optimizer = AdamW(model.parameters(), lr=5e-5)

# --------------------
# 8. Fine-tuning loop
# --------------------
print("\nStarting fine-tuning...")
for i, feature in enumerate(features):
    optimizer.zero_grad()

    outputs = model(
        input_ids=feature["input_ids"],
        attention_mask=feature["attention_mask"],
        start_positions=feature["start_positions"],
        end_positions=feature["end_positions"]
    )

    loss = outputs.loss
    loss.backward()
    optimizer.step()

    print(f"Step {i+1} - Training loss: {loss.item():.4f}")

# --------------------
# 9. Evaluation on a new example
# --------------------
model.eval()

test_context = "Yunnan University is located in Kunming, China."
test_question = "Where is Yunnan University located?"

inputs = tokenizer(test_question, test_context, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits)

predicted_answer = tokenizer.decode(
    inputs["input_ids"][0][start_idx:end_idx + 1],
    skip_special_tokens=True
)

print("\n============================")
print("Evaluation Result")
print("============================")
print("Question:", test_question)
print("Predicted Answer:", predicted_answer)
