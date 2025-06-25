import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up the device (CPU)
device = torch.device("cpu")

# Load the model and tokenizer for Phi-4
model_name = "microsoft/Phi-4"  # Adjust if necessary
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move the model to CPU
model.to(device)

# Input prompt (adjust it according to your use case)
prompt = "Please generate a list of furniture items with their attributes."
# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate output with a reasonable max_length for text generation
output_tokens = model.generate(**inputs, max_length=300)

# Decode the generated tokens back into text
response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

# Print the result
print("Generated output:")
print(response)

# Save the output to a text file
output_file = "generated_output.txt"
with open(output_file, "w") as f:
    f.write(response)
print(f"Output saved to {output_file}")

# List of objects to check
expected_objects = ["sofa", "table", "chair", "bed", "lamp", "bookshelf", "desk"]

# Check how many objects are mentioned in the generated text
found_objects = [obj for obj in expected_objects if obj in response.lower()]
num_found_objects = len(found_objects)

print(f"Number of mentioned objects: {num_found_objects}")
print("Objects found:", found_objects)

from bert_score import score

# Calculate text diversity (unique words proportion)
unique_words = set(response.split())
diversity_score = len(unique_words) / len(response.split())
print(f"Diversity score (proportion of unique words): {diversity_score:.4f}")
