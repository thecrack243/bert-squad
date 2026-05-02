import numpy as np

def build_tokens(sentence):
    tokens = ["[SOS]"] + sentence.split() + ["[EOS]"]
    return tokens

def build_embeddings(tokens):
    embeddings = {}
    for token in tokens:
        token_lower = token.lower()
        if token_lower in ["bank"]:
            embeddings[token] = np.array([0.8, 0.7, 0.6, 0.2])
        elif token_lower in ["money", "cash", "deposit"]:
            embeddings[token] = np.array([0.9, 0.8, 0.7, 0.1])
        elif token_lower in ["river", "stream"]:
            embeddings[token] = np.array([0.9, 0.2, 0.1, 0.8])
        elif token_lower in ["going", "deposited", "is", "hit"]:
            embeddings[token] = np.array([0.2, 0.3, 0.1, 0.0])
        else:
            embeddings[token] = np.random.rand(4) * 0.2
    return embeddings

def compute_raw_scores(query_token, tokens, embeddings):
    raw_scores = []
    q = embeddings[query_token]
    for token in tokens:
        k = embeddings[token]
        score = np.dot(q, k)
        raw_scores.append(score)
    return np.array(raw_scores, dtype=float)

def scale_scores(raw_scores, d_k=64):
    return raw_scores / np.sqrt(d_k)

def softmax(scores):
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / np.sum(exp_scores)

def run_attention_experiment(sentence, query_token):
    print("="*70)
    print(f"Input sentence:\n{sentence}\n")
    
    tokens = build_tokens(sentence)
    embeddings = build_embeddings(tokens)
    
    print(f"Tokens:\n{tokens}\n")
    print(f"Query token: '{query_token}'\n")
    
    raw_scores = compute_raw_scores(query_token, tokens, embeddings)
    print("Step 1: Raw similarity scores (Q·K):")
    for t, s in zip(tokens, raw_scores):
        print(f"{t:10s} -> {s:.4f}")
    print()
    
    scaled_scores = scale_scores(raw_scores)
    print("Step 2: Scaled scores:")
    for t, s in zip(tokens, scaled_scores):
        print(f"{t:10s} -> {s:.4f}")
    print()
    
    attention_weights = softmax(scaled_scores)
    print("Step 3: Attention weights (softmax):")
    for t, s in zip(tokens, attention_weights):
        print(f"{t:10s} -> {s:.4f}")
    print("\n")
    
    most_influential = tokens[np.argmax(attention_weights)]
    print("\nConclusion:")
    print(
        f"The word '{most_influential}' contributes the most to the meaning of '{query_token}'.\n"
    )

sentence_1 = "He deposited money in the bank"
query_token_1 = "bank"
run_attention_experiment(sentence_1, query_token_1)

sentence_2 = "He is going to the river's bank"
query_token_2 = "bank"
run_attention_experiment(sentence_2, query_token_2)

sentence_3 = "He hit the ball with a bat"
query_token_3 = "bat"
run_attention_experiment(sentence_3, query_token_3)
