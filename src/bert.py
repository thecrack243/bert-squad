import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizerFast, BertForQuestionAnswering
from datasets import load_dataset
import re
from collections import Counter

# Step 1: Install and Import Libraries
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

# Step 2: Load SQuAD v2 Dataset
squad = load_dataset("squad_v2")

# Step 3: Tokenization and Preparation of Features
def prepare_features(example):
    tokenized = tokenizer(
        example["question"],
        example["context"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_offsets_mapping=True
    )

    offsets = tokenized["offset_mapping"]
    answers = example["answers"]["text"]

    # Handle unanswerable case
    if len(answers) == 0:
        tokenized["start_positions"] = 0
        tokenized["end_positions"] = 0
        tokenized.pop("offset_mapping")
        return tokenized

    answer_text = answers[0]
    answer_start = example["answers"]["answer_start"][0]
    answer_end = answer_start + len(answer_text)

    start_token = 0
    end_token = 0

    for idx, (start, end) in enumerate(offsets):
        if start <= answer_start < end:
            start_token = idx
        if start < answer_end <= end:
            end_token = idx
            break

    tokenized["start_positions"] = start_token
    tokenized["end_positions"] = end_token
    tokenized.pop("offset_mapping")

    return tokenized

# Apply preprocessing on the dataset
features = squad["train"].select(range(200)).map(prepare_features)
features.set_format(type="torch")

# Step 4: DataLoader and Optimizer
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
loader = DataLoader(features, batch_size=4, collate_fn=data_collator)
optimizer = AdamW(model.parameters(), lr=3e-5)

# Step 5: Evaluation Metrics
def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

def f1_score_single(pred, true):
    pred_tokens = normalize(pred).split()
    true_tokens = normalize(true).split()

    common = Counter(pred_tokens) & Counter(true_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(true_tokens)

    return (2 * precision * recall) / (precision + recall)

def evaluate_model(y_true, y_pred):
    f1_scores = []

    for true, pred in zip(y_true, y_pred):
        f1_scores.append(f1_score_single(pred, true))
    f1 = sum(f1_scores) / len(f1_scores) * 100

    # Status based on F1 score
    if f1 == 100:
        status = "Perfect QA model"
    elif f1 >= 90:
        status = "Very good QA model"
    elif f1 >= 50:
        status = "Good model"
    else:
        status = "Needs improvement"

    return f"F1-score: {f1:.2f}%, Status: {status}"

# Step 6: Training Loop
model.train()

for epoch in range(2):
    for batch in loader:
        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            start_positions=batch["start_positions"],
            end_positions=batch["end_positions"]
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Step 7: Testing
model.eval()

test_context = "Tesla was founded in 2003 and is headquartered in Austin."
test_question = "Where is Tesla headquartered?"

inputs = tokenizer(test_question, test_context, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

start_logits = outputs.start_logits[0]
end_logits = outputs.end_logits[0]

sequence_ids = inputs.sequence_ids()

context_start = sequence_ids.index(1)
context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

# mask logits outside context
start_logits[:context_start] = -1e9
end_logits[:context_start] = -1e9

start_logits[context_end+1:] = -1e9
end_logits[context_end+1:] = -1e9

# pick best span
start_idx = torch.argmax(start_logits)
end_idx = torch.argmax(end_logits)

if end_idx < start_idx:
    end_idx = start_idx

answer_tokens = inputs["input_ids"][0][start_idx:end_idx + 1]

predicted_answer = tokenizer.decode(
    answer_tokens,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True
)

# Step 8: Example Prediction
y_true = ["Austin, Texas", "Kunming"]
y_pred = [predicted_answer, "kunming"]

# Final Evaluation
evaluation = evaluate_model(y_true, y_pred)
print(evaluation)