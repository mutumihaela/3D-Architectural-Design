import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up the device (CPU)
device = torch.device("cpu")

# Load the model and tokenizer for Phi-4 with float32 for CPU compatibility
model_name = "microsoft/Phi-4"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

# Move the model to CPU
model.to(device)

# Improved prompt for better specificity
prompt = (
    "Generate a detailed list of wardrobes. Each wardrobe should include the following attributes: "
    "number of doors, material (e.g., wood, metal, glass), color, dimensions, and an estimated price range. "
    "Provide at least five unique wardrobe options with diverse features."
)

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate output with optimized inference settings
output_tokens = model.generate(
    **inputs, max_length=500, do_sample=True, temperature=0.5, top_k=40
)

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
expected_objects = ["wardrobe", "sofa", "table", "chair", "bed", "lamp", "bookshelf", "desk"]

# Check how many objects are mentioned in the generated text
found_objects = [obj for obj in expected_objects if obj in response.lower()]
num_found_objects = len(found_objects)

print(f"Number of mentioned objects: {num_found_objects}")
print("Objects found:", found_objects)

from bert_score import score

# Calculate text diversity (unique words proportion)
words = response.split()
if words:
    unique_words = set(words)
    diversity_score = len(unique_words) / len(words)
else:
    diversity_score = 0
print(f"Diversity score (proportion of unique words): {diversity_score:.4f}")
