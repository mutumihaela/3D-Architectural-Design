from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

# Inițializare model și tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)


# Prompt pentru generare de atribute
prompt = """
Generează o listă JSON cu 5 obiecte de mobilier 3D.
Asigură-te că output-ul este un JSON valid, fără text suplimentar.
Structura trebuie să fie:
[
    {
        "tip_obiect": "...",
        "material": "...",
        "stil": "...",
        "dimensiuni": {"inaltime": "...", "latime": "...", "adancime": "..."},
        "culoare": "...",
        "textura": "...",
        "mobilitate": "...",
        "nivel_detaliu": "low-poly/mid-poly/high-poly"
    },
    ...
]
"""

# Tokenizare și inferență
inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
output_tokens = model.generate(**inputs, max_length=500)

# Decodificare răspuns
response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

# Validare JSON
try:
    data = json.loads(response)
    print(json.dumps(data, indent=4))
    
    # Metrici simple pentru evaluare
    num_objects = len(data)
    num_valid_objects = sum(1 for obj in data if "tip_obiect" in obj and "material" in obj)
    
    print(f"\nNumăr total de obiecte generate: {num_objects}")
    print(f"Obiecte valide cu atribute corecte: {num_valid_objects}/{num_objects}")
except json.JSONDecodeError:
    print("Eroare: Modelul nu a generat JSON valid.")
