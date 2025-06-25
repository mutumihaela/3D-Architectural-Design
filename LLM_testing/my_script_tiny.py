from transformers import AutoModelForCausalLM, AutoTokenizer

# Inițializare model și tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Prompt pentru generarea listei de atribute
prompt = """
Generează o listă de atribute pentru un obiect de mobilier 3D.
Atributele trebuie să includă detalii despre material, formă, dimensiuni, stil și funcționalitate.
Returnează lista într-un format JSON.
"""

# Tokenizare și inferență
inputs = tokenizer(prompt, return_tensors="pt")
output_tokens = model.generate(**inputs, max_length=200)

# Decodificare răspuns
response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print(response)
