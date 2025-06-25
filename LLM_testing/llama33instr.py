# import torch
# import transformers

# # Model ID
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# # Load the pipeline
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.float16},  # Use float16 for broader GPU support
#     device_map="auto",
# )

# # Chat-style message prompt
# messages = [
#     {"role": "system", "content": "You are a helpful assistant for an interior design company."},
#     {"role": "user", "content": "Please generate a list of furniture items with their attributes."}
# ]

# # Generate output
# print("🟢 Generating response...")
# outputs = pipeline(
#     messages,
#     max_new_tokens=100,
#     do_sample=True,
#     temperature=0.7,
# )
# print("✅ Response generated.")

# # Extract the dictionary from the pipeline output
# response = outputs[0]["generated_text"][-1]  # This is a dict with role/content

# # Access the actual content string
# response_text = response["content"]

# # Print generated output
# print("Generated output:")
# print(response_text)

# # List of expected furniture-related keywords
# expected_objects = [
#     "sofa", "table", "chair", "bed", "lamp", "bookshelf", "desk",
#     "wardrobe", "coffee table", "armchair", "couch", "dining table"
# ]

# # Check which expected objects are mentioned in the response
# found_objects = [obj for obj in expected_objects if obj in response_text.lower()]
# print(f"\nNumărul de obiecte menționate: {len(found_objects)}")
# print("Obiectele găsite:", found_objects)

# # Diversity score calculation
# words = response_text.split()
# unique_words = set(words)
# diversity_score = len(unique_words) / len(words) if words else 0
# print(f"Scorul de diversitate (proporția cuvintelor unice): {diversity_score:.4f}")


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer for Meta-Llama-3-8B-Instruct
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
