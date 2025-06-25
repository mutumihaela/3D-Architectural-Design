from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import re

device = torch.device("cpu")
# Alegem modelul Mistral
model_name = "mistralai/Mistral-7B-Instruct-v0.1"

# Inițializăm tokenizer și model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)
model.to(device)

# Prompt clar pentru model
prompt = """Generează o listă JSON cu 5 obiecte de mobilier 3D. 
Fiecare obiect trebuie să conțină:
- "tip_obiect" (ex: scaun, masă, dulap)
- "material" (ex: lemn, metal, plastic)
- "stil" (ex: modern, clasic, industrial)
- "dimensiuni" (inaltime, latime, adancime în cm)
- "culoare"
- "textura"
- "mobilitate" (fix, pliabil, pe roți)
- "nivel_detaliu" (low-poly, mid-poly, high-poly)

Returnează doar JSON valid, fără text suplimentar."""

# Tokenizare
device = "cuda" if torch.cuda.is_available() else "cpu"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generare output
output_tokens = model.generate(**inputs, max_length=700)
response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

# Validare și curățare JSON
match = re.search(r"\[.*\]", response, re.DOTALL)
if match:
    response = match.group(0)
    try:
        data = json.loads(response)
        print(json.dumps(data, indent=4))  # Afișăm JSON formatat
    except json.JSONDecodeError:
        print("Eroare: JSON invalid, nu a putut fi decodat corect.")
else:
    print("Eroare: Modelul nu a generat JSON valid.")
