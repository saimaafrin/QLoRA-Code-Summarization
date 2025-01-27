import argparse
import pickle as pkl
import pandas as pd
import datasets
import evaluate
from evaluate import load

# Argument parser setup
'''def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Predictions with NLP Metrics")
    parser.add_argument('--predictions_file', type=str, required=True, help="File with predictions")
    parser.add_argument('--test_data_file', type=str, required=True, help="Local test data file path")
    parser.add_argument('--output_file', type=str, required=True, help="Output file to write results")
    return parser.parse_args()'''
# Define the file paths
predictions_file = 'projects/QLoRA/results_CodeLlama-7b-Instruct-hf_both_indication_java_predictions.pkl'
test_data_file = 'projects/QLoRA/results/dataset/java/test.jsonl'
output_file = 'projects/QLoRA/results/csv_files/metrics/metrics_B-M-R-C/mean_scores/metric_cl-7B-qlora_java_test.txt'


# Load predictions and references from local files
def load_data(predictions_file, test_data_file):
    with open(predictions_file, 'rb') as f:
        predictions = pkl.load(f)
    test_data = pd.read_json(test_data_file, lines=True)
    references = [' '.join(doc) for doc in test_data['docstring_tokens']]
    return predictions, references

# Evaluate using multiple metrics
def evaluate_metrics(predictions, references):
    metrics = {
        "bleu": evaluate.load("bleu"),
        "rouge": evaluate.load("rouge"),
        #"bertscore": evaluate.load("bertscore", config_name="roberta-base"),
        "bertscore": evaluate.load("bertscore"),
        "meteor": evaluate.load("meteor"),
        "chrf": evaluate.load("chrf")
    }
    results = {}
    for name, metric in metrics.items():
        if name == "bertscore":
            result = metric.compute(predictions=predictions, references=references, lang="en")
            results[name] = {"f1": sum(result['f1']) / len(references)}
        else:
            result = metric.compute(predictions=predictions, references=references)
            results[name] = {"score": result['score'] if 'score' in result else result}
    return results

    '''all_results = []
    for prediction, reference in zip(predictions, references):
        results = {'prediction': prediction, 'reference': reference}
        for name, metric in metrics.items():
            compute_kwargs = {"lang": "en"} if name == "bertscore" else {}
            result = metric.compute(predictions=[prediction], references=[reference], **compute_kwargs)
            if name == "bertscore":
                results[name] = result['f1'][0]
            else:
                results[name] = result['score'] if 'score' in result else result
        all_results.append(results)
    return all_results'''

    '''
    all_results = []
    for prediction, reference in zip(predictions, references):
        results = {}
        #print("#### checking\n")
        for name, metric in metrics.items():
            #compute_kwargs = {"lang": "en"} if name == "bertscore" else {} 
            compute_kwargs = {model_type="distilbert-base-uncased"} if name == "bertscore" else {}
            result = metric.compute(predictions=[prediction], references=[reference], **compute_kwargs)
            if name == "bertscore":
                results["BERTScore F1"] = result['f1'][0]
            elif name == "rouge":
                results["ROUGE-L"] = result['rougeL']
            elif name == "bleu":
                results["BLEU"] = result['bleu']
            elif name == "meteor":
                results["METEOR"] = result['meteor']
            elif name == "chrf":
                results["ChrF"] = result['score'] if 'score' in result else result
        all_results.append(results)
    print("#### checking-end\n")
    return all_results
    '''
    
# Save evaluation results to a file
def save_results(results, output_file):
    with open(output_file, 'w') as f:
        for name, result in results.items():
            if name == "bertscore":
                f.write(f"{name.upper()} F1: {result['f1']}\n")
            else:
                f.write(f"{name.upper()}: {result['score']}\n")
                
'''def save_results(results, output_file):
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")'''

def main():
    '''args = parse_args()
    predictions, references = load_data(args.predictions_file, args.test_data_file)'''
    predictions, references = load_data(predictions_file, test_data_file)
    results = evaluate_metrics(predictions, references)
    #save_results(results, args.output_file)
    save_results(results, output_file)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
