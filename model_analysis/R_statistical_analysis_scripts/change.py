#### Chnage the column names of csv file ####

import pandas as pd

# Load the CSV file
file_path = 'projects/QLoRA/results/csv_files/metrics/metric_phi-3_qlora_java.csv'
data = pd.read_csv(file_path)

# Modify column names
data = data.rename(columns={"ROUGE-L": "ROUGE_L", "BERTScore F1": "BERTScoreF1"})

# Divide all values of the ChrF column by 100
data['ChrF'] = data['ChrF'] / 100

# Save the modified file
modified_file_path = 'projects/QLoRA/results/csv_files/metrics/metric_phi-3_qlora_java1.csv'
data.to_csv(modified_file_path, index=False)

print(f"Modified file saved at: {modified_file_path}")
