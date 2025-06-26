import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

# Move the model to the selected device
model.to(device)

# Input prompt
prompt = "Create a collection of living room furniture with attributes like style, upholstery type, and dimensions."

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate output with a reasonable max_length for text generation
output_tokens = model.generate(**inputs, max_length=300)

# Decode the generated tokens back into text
response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

# Print the result
print("Generated output:")
print(response)

# List of objects to check
expected_objects = ["sofa", "table", "chair", "bed", "lamp", "bookshelf", "desk", "wardrobe", "coffee table", "armchair", "couch", "dining table"]

# Check how many objects are mentioned in the generated text
found_objects = [obj for obj in expected_objects if obj in response.lower()]
num_found_objects = len(found_objects)

print(f"Number of objects mentioned: {num_found_objects}")
print("Objects found:", found_objects)

# Calculate the diversity of the text (unique words)
unique_words = set(response.split())
diversity_score = len(unique_words) / len(response.split())

print(f"Diversity score (proportion of unique words): {diversity_score:.4f}")
