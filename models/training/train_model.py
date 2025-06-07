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
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import argparse
from huggingface_hub import login, HfFolder
import glob
import gc

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LoRA 설정
LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]

def prepare_dataset(dataset_name, tokenizer, max_seq_len):
    def tokenize_function(examples):
        prompts = []
        for instruction, input_text, output in zip(examples['instruction'], examples['input'], examples['output']):
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                        You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>
                        {instruction}

                        {input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                        {output}<|eot_id|>"""
            prompts.append(prompt)

        return tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_seq_len
        )

    dataset = load_dataset(
        dataset_name
    )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['instruction', 'input', 'output'],
        num_proc=1,
        batch_size=32,
    )

    return tokenized_dataset

def check_huggingface_token():
    token = HfFolder.get_token()
    if token is None:
        login()
    return token

def find_latest_checkpoint(output_dir):
    checkpoints = glob.glob(os.path.join(output_dir, "batch_*"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda x: int(x.split('_')[-1]))

class FineTuner:
    def __init__(self, model_name, dataset_name, output_dir, max_seq_len=128):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.max_seq_len = max_seq_len
        self.tokenizer = None
        self.model = None
        self.dataset = None
        self.latest_checkpoint = find_latest_checkpoint(output_dir)

    def load_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.float16
        )

        if self.latest_checkpoint:
            try:
                config = AutoConfig.from_pretrained(self.model_name)
                checkpoint_config = AutoConfig.from_pretrained(self.latest_checkpoint)
                checkpoint_config.model_type = config.model_type
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.latest_checkpoint,
                    config=checkpoint_config,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    quantization_config=bnb_config
                )
            except:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    quantization_config=bnb_config
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                quantization_config=bnb_config
            )
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

    def load_and_process_dataset(self):
        self.dataset = prepare_dataset(
            dataset_name=self.dataset_name,
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len
        )

    def setup_training_args(self, args, batch_start_idx):
        return TrainingArguments(
            output_dir=os.path.join(args.output_dir, f"batch_{batch_start_idx}"),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            optim=args.optim,
            logging_steps=args.logging_steps,
            learning_rate=args.learning_rate,
            fp16=True,
            max_grad_norm=args.max_grad_norm,
            max_steps=args.max_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.num_train_epochs,
            lr_scheduler_type=args.lr_scheduler_type,
            group_by_length=False,
            gradient_checkpointing=True,
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
            torch_compile=False,
            report_to="none",
            load_best_model_at_end=False,
            save_total_limit=2
        )

    def train(self, args):
        check_huggingface_token()
        self.load_model_and_tokenizer()
        self.load_and_process_dataset()

        trainer = Trainer(
            model=self.model,
            args=self.setup_training_args(args, 0),
            train_dataset=self.dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=False)
        )

        trainer.train()
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--model-name', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=128)
    parser.add_argument('--optim', type=str, default="adamw_torch")
    parser.add_argument('--logging-steps', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--max-grad-norm', type=float, default=0.3)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--warmup-ratio', type=float, default=0.1)
    parser.add_argument('--num-train-epochs', type=int, default=3)
    parser.add_argument('--lr-scheduler-type', type=str, default="cosine")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Fine-tuning requires GPU.")
        exit(1)

    finetuner = FineTuner(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir
    )
    finetuner.train(args)

if __name__ == '__main__':
    main()
