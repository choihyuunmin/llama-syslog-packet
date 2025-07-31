import logging
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    AutoConfig
)
from datasets import Dataset
import argparse
from huggingface_hub import login, HfFolder
import glob
import gc
import json
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import pandas as pd

# TOKENIZERS_PARALLELISM 환경변수 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LoRA 설정
LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]

def load_jsonl_files_from_dir(dataset_dir):
    all_data = []
    for file_path in glob.glob(os.path.join(dataset_dir, "*.json")):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)

    return all_data

def prepare_dataset(dataset_dir, tokenizer, max_seq_len=2048):
    def generate_and_tokenize_prompt(examples):
        alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]

        EOS_TOKEN = tokenizer.eos_token
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)

        # 실제로 tokenization 수행
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length", 
            max_length=max_seq_len,
            return_tensors=None
        )
        
        # labels를 input_ids와 동일하게 설정 (causal language modeling을 위해)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized


    all_data = load_jsonl_files_from_dir(dataset_dir)
    
    processed_data = []
    for item in all_data:
        processed_item = {
            "instruction": item["instruction"],
            "input": json.dumps(item["input"], ensure_ascii=False),
            "output": item["output"]
        }
        processed_data.append(processed_item)
    
    dataset = Dataset.from_list(processed_data)

    train_val = dataset.train_test_split(test_size=0.8, shuffle=True, seed=42)

    train_data = train_val["train"].map(
        generate_and_tokenize_prompt,
        batched=True,
        batch_size=32,
        remove_columns=dataset.column_names
    )
    
    val_data = train_val["test"].map(
        generate_and_tokenize_prompt,
        batched=True,
        batch_size=32,
        remove_columns=dataset.column_names
    )

    return train_data, val_data

def check_huggingface_token():
    token = HfFolder.get_token()
    if token is None:
        login()
    return token

def merge_and_upload_to_hf(model, tokenizer, output_dir, hf_model_name, push_to_hub=False):
    logger.info("##10. Merging LoRA adapters with base model")
    
    # LoRA 어댑터를 base model에 merge
    merged_model = model.merge_and_unload()
    
    # 메모리 정리
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # 로컬에 저장
    merged_output_dir = os.path.join(output_dir, f"{hf_model_name}_merged")
    os.makedirs(merged_output_dir, exist_ok=True)
    
    logger.info(f"##11. Saving merged model to {merged_output_dir}")
    merged_model.save_pretrained(merged_output_dir)
    tokenizer.save_pretrained(merged_output_dir)
    
    # 허깅페이스에 업로드
    if push_to_hub:
        try:
            logger.info(f"##12. Uploading merged model to Hugging Face: {hf_model_name}")
            merged_model.push_to_hub(hf_model_name, use_temp_dir=True)
            tokenizer.push_to_hub(hf_model_name, use_temp_dir=True)
            logger.info(f"##13. Successfully uploaded to https://huggingface.co/{hf_model_name}")
        except Exception as e:
            logger.error(f"Failed to upload to Hugging Face: {e}")
            logger.info("Model saved locally only.")
    
    return merged_model, merged_output_dir

class FineTuner:
    def __init__(self, model_name, dataset_dir, output_dir, max_seq_len=2048):
        self.model_name = model_name
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.max_seq_len = max_seq_len
        self.tokenizer = None
        self.model = None

    def load_model_and_tokenizer(self):
        logger.info("##2. Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        logger.info("##3. Loading model")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.float16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=bnb_config
        )

        self.config = AutoConfig.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        logger.info("##5. Setting use_cache to False")
        self.model.config.use_cache = False
        self.model = prepare_model_for_kbit_training(self.model)

        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
        logger.info("##6. model loaded")

    def load_and_process_dataset(self):
        logger.info("##7. Loading and processing dataset")
        train_data, val_data = prepare_dataset(
            dataset_dir=self.dataset_dir,
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len
        )
        
        logger.info("##8. Dataset loaded")
        return train_data, val_data

    def train(self, args):
        check_huggingface_token()
        self.load_model_and_tokenizer()
        train_data, val_data = self.load_and_process_dataset()
        logger.info("##9. Training")

        training_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir, args.hf_model_name),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            optim=args.optim,
            logging_steps=args.logging_steps,
            learning_rate=args.learning_rate,
            fp16=True,
            max_grad_norm=args.max_grad_norm,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.num_train_epochs,
            lr_scheduler_type=args.lr_scheduler_type,
            group_by_length=True,
            gradient_checkpointing=True,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            torch_compile=False,
            label_names=["labels"],
            eval_steps=args.logging_steps,
            report_to="none"
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=None,
            return_tensors="pt"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            data_collator=data_collator
        )

        trainer.train()

        self.model.save_pretrained(os.path.join(args.output_dir, args.hf_model_name))
        self.config.save_pretrained(os.path.join(args.output_dir, args.hf_model_name))

        merged_model, merged_output_dir = merge_and_upload_to_hf(
            self.model,
            self.tokenizer,
            args.output_dir,
            args.hf_model_name,
            push_to_hub=True
        )

        return merged_model, merged_output_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, required=True)
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--hf-model-name', type=str, required=True, help="Hugging Face model name to upload")
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4)
    parser.add_argument('--optim', type=str, default="adamw_torch")
    parser.add_argument('--logging-steps', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--max-grad-norm', type=float, default=0.3)
    parser.add_argument('--warmup-ratio', type=float, default=0.1)
    parser.add_argument('--num-train-epochs', type=int, default=3)
    parser.add_argument('--lr-scheduler-type', type=str, default="cosine")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Fine-tuning requires GPU.")
        exit(1)

    finetuner = FineTuner(
        model_name=args.model_name,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir
    )
    merged_model, merged_output_dir = finetuner.train(args)

if __name__ == '__main__':
    main()