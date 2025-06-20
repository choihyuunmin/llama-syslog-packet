from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("choihyuunmin/LlamaTrace")
tokenizer = AutoTokenizer.from_pretrained("choihyuunmin/LlamaTrace")

input_text = "hello, what is your name?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
