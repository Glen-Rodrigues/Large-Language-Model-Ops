from transformers import AutoModel, AutoTokenizer

# Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

# Load a language model
model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall")

# Tokenize the sentence
tokens = tokenizer('Hello World', return_tensors='pt')

# Process the tokens
output = model(**tokens)[0]

print("********* OUTPUT - Torch Size *****")
print("Torch Size\n", output.shape)

for token in tokens['input_ids'][0]:
    print(tokenizer.decode(token))

print("********* OUTPUT - Tensor *******\n", output)