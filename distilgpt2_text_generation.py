#Bigin of Simple T-I-T-O Transformers program
from transformers import pipeline

# Create a text generation pipeline
generator = pipeline(
    "text-generation",
    model="distilgpt2"
)
# Give a prompt - Input text
prompt = "Artificial Intelligence is a "

# Generate text
result = generator(
    prompt,
    max_new_tokens=30,
    do_sample=True,
    temperature=0.7
)

# Display output
print("\n*** Input Text ***")
print(prompt)

print("\n*** Generated Text ***")
print(result[0]["generated_text"])

#End of T-I-T-O Transformers program
