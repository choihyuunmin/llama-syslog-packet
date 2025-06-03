import json
import logging
import os
from pathlib import Path
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
import argparse
from huggingface_hub import login, HfFolder

# 메모리 최적화 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
torch.cuda.empty_cache()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LoRA 설정
LORA_R = 8  # 메모리 효율을 위해 랭크 감소
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

def check_huggingface_token():
    """Check Hugging Face token."""
    token = HfFolder.get_token()
    if token is None:
        print("Please input your Hugging Face token.")
        token = input("Hugging Face 토큰을 입력하세요: ")
        login(token)
    return token

class FineTuner:
    def __init__(self, model_name, dataset_path, output_dir, max_seq_len=512):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.max_seq_len = max_seq_len
        self.tokenizer = None
        self.model = None
        self.dataset = None

    def load_model_and_tokenizer(self):
        logger.info(f"Loading tokenizer for {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        logger.info(f"Loading model {self.model_name}...")
        
        # 4비트 양자화 설정
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        if torch.cuda.is_available():
            # 사용 가능한 GPU 확인
            available_gpus = list(range(torch.cuda.device_count()))
            logger.info(f"Available GPUs: {available_gpus}")
            for gpu_id in available_gpus:
                logger.info(f"GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
                logger.info(f"GPU {gpu_id} Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.2f} GB")
            
            # 메모리 캐시 정리
            torch.cuda.empty_cache()
            
            # 모델을 CPU에 먼저 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="cpu",
                quantization_config=bnb_config,
                low_cpu_mem_usage=True
            )
            
            # LoRA 설정 적용
            self.model = prepare_model_for_kbit_training(self.model)
            lora_config = LoraConfig(
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                target_modules=LORA_TARGET_MODULES,
                lora_dropout=LORA_DROPOUT,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
            
            # DataParallel 설정
            self.model = nn.DataParallel(self.model, device_ids=available_gpus)
            self.model = self.model.cuda()
            
            # 그래디언트 체크포인팅 활성화
            self.model.module.config.use_cache = False
            self.model.module.gradient_checkpointing_enable()
            
            logger.info(f"Model distributed across {len(available_gpus)} GPUs")
            self.model.print_trainable_parameters()
        else:
            raise RuntimeError("GPU가 필요합니다. CUDA를 사용할 수 없습니다.")

    def load_and_process_dataset(self):
        logger.info(f"Loading dataset from {self.dataset_path}...")
        dataset = []
        try:
            for json_file in Path(self.dataset_path).glob('*.json'):
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        logger.warning(f"Skipping {json_file}: Expected list format")
                        continue
                    dataset.extend(data)
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

        if not dataset:
            raise ValueError("No valid data loaded from JSON files")

        self.dataset = Dataset.from_list(dataset)

        def format_prompt(item):
            category = item.get('category', 'general')
            prompt = f"""### Instruction:
                        You are a helpful assistant expert in network and system log analysis.
                        You are given a log file and a question. Please provide a detailed and accurate answer.

                        ### Category:
                        {category}

                        ### Question:
                        {item['question']}

                        ### Answer:
                        {item['answer']}

                        ### End"""
            return {"text": prompt}

        self.dataset = self.dataset.map(format_prompt)

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="pt"
            )

        self.dataset = self.dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text", "question", "answer", "category"]
        )

    def setup_training_args(self, args):
        return TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=128,  # 메모리 효율을 위해 증가
            optim=args.optim,
            logging_steps=args.logging_steps,
            learning_rate=args.learning_rate,
            fp16=True,  # FP16 활성화
            max_grad_norm=args.max_grad_norm,
            max_steps=args.max_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.num_train_epochs,
            lr_scheduler_type=args.lr_scheduler_type,
            group_by_length=True,
            gradient_checkpointing=True,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            torch_compile=False,
            deepspeed=None,
            ddp_find_unused_parameters=False,
            local_rank=-1,
            no_cuda=False,
            dataloader_drop_last=True,
            remove_unused_columns=True,
            report_to="none",
            load_best_model_at_end=False,
            save_total_limit=0
        )

    def train(self, args):
        try:
            check_huggingface_token()
            self.load_model_and_tokenizer()
            self.load_and_process_dataset()

            training_args = self.setup_training_args(args)
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset,
                data_collator=data_collator
            )

            logger.info("Starting fine-tuning...")
            trainer.train()
            
            logger.info("Saving fine-tuned model...")
            trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)
            
        except Exception as e:
            logger.error(f"모델 학습 중 오류 발생: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Train a model on network and system log analysis')
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--model-name', type=str, required=True, help='Name of the model to train')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the trained model')
    parser.add_argument('--batch-size', type=int, default=1, help='Training batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=128, help='Gradient accumulation steps')
    parser.add_argument('--optim', type=str, default="adamw_torch", help='Optimizer')
    parser.add_argument('--logging-steps', type=int, default=100, help='Logging steps')
    parser.add_argument('--learning-rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--max-grad-norm', type=float, default=0.3, help='Maximum gradient norm')
    parser.add_argument('--max-steps', type=int, default=-1, help='Maximum training steps')
    parser.add_argument('--warmup-ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--num-train-epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--lr-scheduler-type', type=str, default="cosine", help='Learning rate scheduler type')
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Fine-tuning requires GPU.")
        exit(1)
        
    finetuner = FineTuner(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir
    )
    
    finetuner.train(args)

if __name__ == '__main__':
    main()
