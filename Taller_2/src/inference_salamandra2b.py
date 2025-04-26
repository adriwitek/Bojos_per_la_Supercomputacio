from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch


MODEL_ID = "salamandra-2b-instruct"



# text contiene el texto "limpio" que queremos pasarle al modelo
text = '¿Quién escribió Cien años de soledad y en qué año se publicó por primera vez?'



# Cargamos el modelo y el tokenizador 
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16
  )

# Preparamos el mensaje formateandolo acorde a la Chat Template
message = [ { "role": "user", "content": text } ]
date_string = datetime.today().strftime('%Y-%m-%d')
prompt = tokenizer.apply_chat_template(
    message,
    tokenize=False,
    add_generation_prompt=True,
    date_string=date_string
)

# Tokenizamos
inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

# Hacemos inferencia con el modelo
outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=200)

# Imprimimos la respuesta del modelo (podemos ignorar los warnings)
print("\n" * 2)
print("-"*60)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

