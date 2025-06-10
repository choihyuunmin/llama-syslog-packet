import json
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import openai
from typing import Optional, Union, List, Dict
import time
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_benchmark_data(benchmark_path: Path) -> List[Dict]:
    """Load benchmark data from JSON file."""
    with open(benchmark_path, 'r') as f:
        return json.load(f)

def save_predictions(predictions: List[Dict], output_path: Path):
    """Save model predictions to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)

def generate_openai_prediction(
    prompt: str,
    model: str,
    max_tokens: int = 1000,
    temperature: float = 0.7
) -> str:
    """Generate prediction using OpenAI API."""
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a network security expert. Analyze the given network traffic and provide detailed insights."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error generating prediction with OpenAI: {str(e)}")
        return ""

def generate_huggingface_prediction(
    prompt: str,
    model_name: str,
    max_length: int = 1000,
    temperature: float = 0.7
) -> str:
    """Generate prediction using Hugging Face model."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logging.error(f"Error generating prediction with Hugging Face: {str(e)}")
        return ""

def generate_predictions(
    benchmark_data: List[Dict],
    model: str,
    model_type: str = "openai",
    max_tokens: int = 1000,
    temperature: float = 0.7
) -> List[Dict]:
    """Generate predictions for all benchmark cases."""
    predictions = []
    
    for case in benchmark_data:
        prompt = case['input']
        start_time = time.time()
        
        if model_type == "openai":
            prediction = generate_openai_prediction(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )
        else:  # huggingface
            prediction = generate_huggingface_prediction(
                prompt=prompt,
                model_name=model,
                max_length=max_tokens,
                temperature=temperature
            )
        
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        predictions.append({
            'case_id': case['id'],
            'prompt': prompt,
            'prediction': prediction,
            'execution_time_ms': execution_time
        })
        
        logging.info(f"Generated prediction for case {case['id']}")
    
    return predictions

def main():
    parser = argparse.ArgumentParser(description='Generate model predictions for benchmark')
    parser.add_argument('--model', type=str, required=True,
                      help='Model name')
    parser.add_argument('--model-type', type=str, default='openai',
                      choices=['openai', 'huggingface'],
                      help='Type of model to use')
    parser.add_argument('--benchmark-dir', type=str, default='benchmark_data',
                      help='Directory containing benchmark data')
    parser.add_argument('--output-dir', type=str, default='model_predictions',
                      help='Directory to save predictions')
    parser.add_argument('--max-tokens', type=int, default=1000,
                      help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Temperature for generation')
    
    args = parser.parse_args()
    setup_logging()
    
    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    
    # Load benchmark data
    benchmark_data = load_benchmark_data(benchmark_dir / 'benchmark.json')
    
    model_name = args.model
    # Generate predictions
    predictions = generate_predictions(
        benchmark_data=benchmark_data,
        model=model_name,
        model_type=args.model_type,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    model_name = model_name.replace('/', '_')
    # Save predictions
    output_path = output_dir / f'{model_name}_predictions.json'
    save_predictions(predictions, output_path)
    
    logging.info(f"Predictions saved to {output_path}")

if __name__ == '__main__':
    main() 