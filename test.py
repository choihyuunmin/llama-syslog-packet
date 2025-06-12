from transformers import AutoModelForCausalLM

model_name = "choihyuunmin/LlamaTrace"
model = AutoModelForCausalLM.from_pretrained(model_name)

print(model)