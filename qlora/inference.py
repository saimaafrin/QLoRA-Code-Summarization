import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np
import random
from datasets import Dataset
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, Dataset
import pandas as pd
import evaluate
import torch
import nltk
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
import argparse
import numpy as np
from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from trl import AutoModelForCausalLMWithValueHead
from transformers import TrainingArguments, Trainer
from trl import SFTTrainer
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import datasets
import re
import pickle as pkl
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset


# Argument parser setup
def parse_args():
    parser = argparse.ArgumentParser(description="Run N-Gram Language Model Evaluation")
    parser.add_argument('--use_qlora', action='store_true', help="Set to True to use QLoRA")
    parser.add_argument('--language', type=str, default='python', help="Programming language")
    parser.add_argument('--base_model_name', type=str, default='deepseek-ai/deepseek-coder-33b-instruct', help="Base model name")
    parser.add_argument('--lora_ckpt', type=str, default='results/deepseek-coder-33b-instruct_both_indication_5k/', help="LoRA checkpoint path")
    parser.add_argument('--output_file', type=str, help="Output file to write results", required=True)
    args = parser.parse_args()
    return args


# Main function
def main():
    args = parse_args()
    save_path = '_'.join(args.lora_ckpt.split('/')[-2:]) + f'_{args.language}'
    print("Saving to", save_path)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        cache_dir='./models',
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    if args.use_qlora:
        model = PeftModel.from_pretrained(model, args.lora_ckpt)
        print('Finished loading QLoRA')

    # Load dataset
    ds = load_dataset("google/code_x_glue_ct_code_to_text", args.language, cache_dir='datasets')
    test_dataset = ds["test"]
#     test_dataset = test_dataset.shuffle().select(range(5000))

    # Apply chat template
#     def apply_chat_template(example):
#         def remove_comments_from_code(code):
#             code_without_comments = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
#             return code_without_comments.strip()

#         code = example['code']
#         chat = [{"role": "system", "content": "Provide summarization to given code segment"},
#                 {"role": "user", "content": f"Summarize:\n{remove_comments_from_code(code)}"}]
#         example["text"] = tokenizer.apply_chat_template(
#             chat, tokenize=False, add_generation_prompt=True
#         )
#         return example


    def apply_chat_template(example):
        def remove_comments_from_code(code):
            code_without_comments = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)

            # Return the cleaned code
            return code_without_comments.strip()

#         code = remove_comments_from_code(example['code'])
        code = ' '.join(example['code_tokens'])
        language = example['language']
        chat = [{"role": "system", "content": "Provide summarization to given code segment"},
          {"role": "user", "content": f"Summarize {language} code:\n{code}"},
        ]
        if tokenizer.chat_template == None:
            tokenizer.chat_template = "{%- for message in messages -%}\n{% if message['role'] == 'system' %}{% if loop.first %}{{ message['content'] }}{% endif %}{% elif message['role'] == 'user' %}{{ 'Human: ' + message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + '\n\n' }}{% endif %}\n{%- endfor -%}\n{% if add_generation_prompt %}Human: {{ '' }}{% endif %}"

        example["text"] = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        return example
    
    column_names = list(test_dataset.features)
    processed_test_dataset = test_dataset.map(
        apply_chat_template,
        num_proc=10,
        remove_columns=column_names,
        desc="Applying chat template to test set"
    )
    print(processed_test_dataset[0]['text'])
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    print(tokenizer.padding_side)
    model.generation_config.temperature = 0
    model.generation_config.do_sample = False
    model.generation_config.num_beams = 1

    # Generate predictions
    outs = []
    pipe = transformers.pipeline(task='text-generation', model=model, tokenizer=tokenizer, max_new_tokens=25)
    for out in tqdm(pipe(KeyDataset(processed_test_dataset, "text"), batch_size=16, return_full_text=True, truncation="only_first", num_beams=1)):
        outs.append(out)

    predictions = [i[0]['generated_text'].split('SUMMARY:')[-1].split('DONE')[0].strip() for i in outs]

    with open(f'preds/{save_path}_predictions.pkl', 'wb') as f:
        pkl.dump(predictions, f)

    if not args.use_qlora:
        predictions = [i.split('<|assistant|>')[-1].strip() for i in predictions]
        predictions = [i.split('[/INST]')[-1].strip() for i in predictions]

    # Evaluate using multiple metrics
    references = [' '.join(i) for i in test_dataset['docstring_tokens']]
    
    sacrebleu = evaluate.load("bleu")
    bleu_results = sacrebleu.compute(predictions=predictions, references=references)

    rouge = evaluate.load("rouge")
    rouge_results = rouge.compute(predictions=predictions, references=references)

    bertscore = evaluate.load("bertscore")
    bert_results = bertscore.compute(predictions=predictions, references=references, model_type="roberta-large")
    bert_f1 = sum(bert_results['f1']) / len(references)

    meteor = evaluate.load('meteor')
    meteor_results = meteor.compute(predictions=predictions, references=references)

    chrf = evaluate.load("chrf")
    chrf_results = chrf.compute(predictions=predictions, references=references)

    # Save results to a file
    
    with open(f'inference_results/{save_path}_results.txt', 'w') as f:
        f.write(f"BLEU: {bleu_results['bleu']}\n")
        f.write(f"ROUGE-L: {rouge_results['rougeL']}\n")
        f.write(f"BERTScore F1: {bert_f1}\n")
        f.write(f"METEOR: {meteor_results['meteor']}\n")
        f.write(f"ChrF: {chrf_results['score']}\n")

    print("Results saved to file.")


if __name__ == "__main__":
    main()
