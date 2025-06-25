# phi2_promptgen.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ✅ Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Phi-2 model (best mix of quality and speed)
model_name = "microsoft/phi-2"
print(f"🚀 Loading {model_name} on {device}...")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
model.to(device)
print("✅ Phi-2 loaded and ready.")

def generate_prompt(class_name: str) -> str:
    # ✅ Use instruction-tuned format to boost response quality
    prompt = (
        f"Instruct: Describe a high-quality 3D model of a {class_name}. Include:\n"
        f"- Material (e.g., wood, metal, plastic)\n"
        f"- Style (e.g., modern, classic)\n"
        f"- Shape and dimensions\n"
        f"- Typical use or placement\n"
        f"Output:"
    )

    # Tokenize and move to GPU or CPU
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate output
    output_tokens = model.generate(
        **inputs,
        max_length= 200,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id 
    )

    # Decode the response
    response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    # Extract the text after "Output:" (if present)
    if "Output:" in response:
        response = response.split("Output:")[-1].strip()

    # Fallback if empty
    if not response.strip():
        response = f"A {class_name} with standard furniture characteristics."

    return response.strip()

