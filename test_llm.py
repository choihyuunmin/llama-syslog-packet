from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 모델과 토크나이저 불러오기
model_name = "choihyuunmin/LlamaTrace"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 입력 질문
question = "hello, what is your name?"
inputs = tokenizer(question, return_tensors="pt")

# GPU 사용 가능 시 활용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

# 응답 생성
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,       # (선택) 더 다양하게 응답받고 싶을 때 사용
        temperature=0.7,      # (선택) 창의성 조절
        top_p=0.9             # (선택) nucleus sampling
    )

# 결과 출력
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("=== 질문 ===")
print(question)
print("=== 응답 ===")
print(response)