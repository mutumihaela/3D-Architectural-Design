import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bert_score import score

# Set up the device (CPU)
device = torch.device("cpu")

# Load the model and tokenizer for Microsoft Phi-4
model_name = "microsoft/phi-4"  # Adjusted to Microsoft Phi-4 model
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

# Listă de obiecte de verificat
expected_objects = ["sofa", "table", "chair", "bed", "lamp", "bookshelf", "desk"]

# Verificăm câte obiecte sunt menționate în textul generat
found_objects = [obj for obj in expected_objects if obj in response.lower()]
num_found_objects = len(found_objects)

print(f"Numărul de obiecte menționate: {num_found_objects}")
print("Obiectele găsite:", found_objects)

# Calculăm diversitatea textului (cuvinte unice)
unique_words = set(response.split())
diversity_score = len(unique_words) / len(response.split())
print(f"Scorul de diversitate (proporția cuvintelor unice): {diversity_score:.4f}")

