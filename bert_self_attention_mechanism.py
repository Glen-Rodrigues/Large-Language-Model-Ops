import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# 1. Load Hugging Face tokenizer and model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Use evaluation mode
model.eval()

# 2. Input sentence
sentence = "The dog chased the squirrel because it was scared."
inputs = tokenizer(
    sentence,
    return_tensors="pt"
)

print("Tokens:")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print(tokens)

# 3. Get token embeddings from BERT
with torch.no_grad():
    outputs = model(**inputs)

# Last hidden state
# Shape: [batch_size, number_of_tokens, hidden_size]

X = outputs.last_hidden_state

print("\nInput representation shape:")
print(X.shape)

# 4. Create Q, K and V
hidden_size = X.shape[-1]

# Three learnable transformations
W_Q = nn.Linear(hidden_size, hidden_size, bias=False)
W_K = nn.Linear(hidden_size, hidden_size, bias=False)
W_V = nn.Linear(hidden_size, hidden_size, bias=False)

# Generate Q, K and V
Q = W_Q(X)
K = W_K(X)
V = W_V(X)

print("\nQ shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)

# 5. Calculate attention scores
# Q @ K^T
scores = torch.matmul(Q, K.transpose(-2, -1))

# Scale by sqrt(d_k)
scores = scores / torch.sqrt(
    torch.tensor(hidden_size, dtype=torch.float32)
)

print("\nAttention score shape:")
print(scores.shape)

# 6. Apply Softmax
attention_weights = torch.softmax(scores, dim=-1)

print("\nAttention weights shape:")
print(attention_weights.shape)

# 7. Calculate contextual representation
context = torch.matmul(attention_weights, V)

print("\nContext representation shape:")
print(context.shape)

# 8. Display attention for each token
print("\nAttention Matrix:\n")

for i, token in enumerate(tokens):
    print(f"\nToken: {token}")
    for j, attended_token in enumerate(tokens):
        weight = attention_weights[0, i, j].item()
        if weight > 0.05:
            print(
                f"  {attended_token:12s} -> {weight:.4f}"
            )
