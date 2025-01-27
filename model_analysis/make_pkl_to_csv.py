import pandas as pd

# Paths to the files
predictions_path = 'projects/QLoRA/results/model_prediction_files/CodeLlama-34b-Instruct-hf_both_indication_checkpoint-25000_java_predictions.pkl'
test_jsonl_path = 'projects/QLoRA/results/dataset/java/test.jsonl'

# Load the predictions and test data
predictions = pd.read_pickle(predictions_path)
test_data = pd.read_json(test_jsonl_path, lines=True)

# Concatenate 'docstring_tokens' into a single string and convert to lowercase
test_data['docstring'] = test_data['docstring_tokens'].apply(lambda tokens: ' '.join(tokens).lower())

# Convert predictions to lowercase
predictions = [pred.lower() for pred in predictions]

# Create a DataFrame for comparison
comparison_df = pd.DataFrame({
    "Model Input": test_data['code'],
    "Model Output (Ground Truth)": test_data['docstring'],
    "Prediction": predictions,
    "Is_perfect": test_data['docstring'] == predictions
})

# Add empty columns for metrics to be filled elsewhere
metrics_columns = ["BLEU", "ROUGE_L", "BERTScoreF1", "METEOR", "ChrF"]
for metric in metrics_columns:
    comparison_df[metric] = None

# Save the DataFrame as a CSV file
csv_file_path = 'projects/QLoRA/results/csv_files/output_csv_files/CL-34B_java.csv'
comparison_df.to_csv(csv_file_path, index=False)

print(f"CSV file saved to {csv_file_path}")

