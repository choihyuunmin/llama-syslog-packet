import json
import logging
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import argparse
from huggingface_hub import login, HfFolder

logger = logging.getLogger(__name__)

def check_huggingface_token():
    """check huggingface token."""
    token = HfFolder.get_token()
    if token is None:
        print("Please input your huggingface token.")
        token = input("Hugging Face 토큰을 입력하세요: ")
        login(token)
    return token

class DatasetGenerator:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        
    def load_json_files(self) -> list:
        """load json files."""
        dataset = []
        for json_file in self.dataset_path.glob('*.json'):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                dataset.extend(data)
        return dataset
        
    def format_prompt(self, item: dict) -> str:
        """generate prompt."""
        return f"Question: {item['question']}\nAnswer: {item['answer']}"
        
    def generate_dataset(self) -> list:
        """generate dataset."""
        raw_data = self.load_json_files()
        return [self.format_prompt(item) for item in raw_data]

def train_model(
    dataset_path: str,
    model_name: str,
    output_dir: str,
    batch_size: int = 4,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    max_grad_norm: float = 1.0,
    warmup_ratio: float = 0.1,
    logging_steps: int = 100,
    save_steps: int = 500,
    eval_steps: int = 500,
    gradient_accumulation_steps: int = 4,
    max_steps: int = -1,
    lr_scheduler_type: str = "cosine",
    optim: str = "adamw_torch",
    weight_decay: float = 0.01,
    fp16: bool = True,
    group_by_length: bool = True,
    report_to: str = "tensorboard",
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "loss",
    greater_is_better: bool = False,
    save_total_limit: int = 2,
    seed: int = 42,
    dataloader_num_workers: int = 4,
    dataloader_pin_memory: bool = True,
    remove_unused_columns: bool = True,
    label_smoothing_factor: float = 0.0,
    max_length: int = 512,
    padding: str = "max_length",
    truncation: bool = True
):
    """모델을 학습시킵니다."""
    try:
        # Hugging Face 토큰 확인
        check_huggingface_token()
        
        # 데이터셋 생성
        dataset_generator = DatasetGenerator(dataset_path)
        dataset = dataset_generator.generate_dataset()
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # 데이터셋 토큰화
        def tokenize_function(examples):
            return tokenizer(
                examples,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors="pt"
            )
            
        tokenized_dataset = tokenize_function(dataset)
        
        # 모델 로드
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # 학습 인자 설정
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            fp16=fp16,
            max_grad_norm=max_grad_norm,
            max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            group_by_length=group_by_length,
            lr_scheduler_type=lr_scheduler_type,
            weight_decay=weight_decay,
            report_to=report_to,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            save_total_limit=save_total_limit,
            seed=seed,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_pin_memory=dataloader_pin_memory,
            remove_unused_columns=remove_unused_columns,
            label_smoothing_factor=label_smoothing_factor,
            save_steps=save_steps,
            eval_steps=eval_steps,
            num_train_epochs=epochs
        )
        
        # 데이터 콜레이터 설정
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # 트레이너 초기화
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator
        )
        
        # 학습 시작
        trainer.train()
        
        # 모델 저장
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
    except Exception as e:
        logger.error(f"모델 학습 중 오류 발생: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Train a model on network and system log analysis')
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--model-name', type=str, required=True, help='Name of the model to train')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the trained model')
    parser.add_argument('--batch-size', type=int, default=4, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max-grad-norm', type=float, default=1.0, help='Maximum gradient norm')
    parser.add_argument('--warmup-ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--logging-steps', type=int, default=100, help='Logging steps')
    parser.add_argument('--save-steps', type=int, default=500, help='Save steps')
    parser.add_argument('--eval-steps', type=int, default=500, help='Evaluation steps')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--max-steps', type=int, default=-1, help='Maximum training steps')
    parser.add_argument('--lr-scheduler-type', type=str, default="cosine", help='Learning rate scheduler type')
    parser.add_argument('--optim', type=str, default="adamw_torch", help='Optimizer')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 training')
    parser.add_argument('--group-by-length', action='store_true', help='Group sequences by length')
    parser.add_argument('--report-to', type=str, default="tensorboard", help='Reporting tool')
    parser.add_argument('--load-best-model-at-end', action='store_true', help='Load best model at end')
    parser.add_argument('--metric-for-best-model', type=str, default="loss", help='Metric for best model')
    parser.add_argument('--greater-is-better', action='store_true', help='Greater is better for metric')
    parser.add_argument('--save-total-limit', type=int, default=2, help='Total number of checkpoints to save')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dataloader-num-workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--dataloader-pin-memory', action='store_true', help='Pin memory for dataloader')
    parser.add_argument('--remove-unused-columns', action='store_true', help='Remove unused columns')
    parser.add_argument('--label-smoothing-factor', type=float, default=0.0, help='Label smoothing factor')
    parser.add_argument('--max-length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--padding', type=str, default="max_length", help='Padding strategy')
    parser.add_argument('--truncation', action='store_true', help='Enable truncation')
    
    args = parser.parse_args()
    
    train_model(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        group_by_length=args.group_by_length,
        report_to=args.report_to,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
        remove_unused_columns=args.remove_unused_columns,
        label_smoothing_factor=args.label_smoothing_factor,
        max_length=args.max_length,
        padding=args.padding,
        truncation=args.truncation
    )

if __name__ == '__main__':
    main() 