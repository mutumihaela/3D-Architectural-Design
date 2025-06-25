import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up the device (CPU)
device = torch.device("cpu")

# Load the model and tokenizer for Phi-4
model_name = "microsoft/phi-4"  # Adjust the model name if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move the model to CPU
model.to(device)

# Input prompt (adjust it according to your use case)
prompt = "Please generate a list of furniture items with their attributes."
# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate output with a larger max_length and sampling enabled
output_tokens = model.generate(
    **inputs, 
    max_length=250, 
    do_sample=True, 
    top_k=50, 
    top_p=0.95, 
    temperature=0.7
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
