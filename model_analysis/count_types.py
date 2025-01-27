import pandas as pd

# Load the data from the uploaded CSV file
file_path = 'projects/QLoRA/results/csv_files/statistical_analysis/Experiments-Tracker - Conflict_Solved_CodeLlama-34B-qlora-python.csv'
data = pd.read_csv(file_path)

# Count the occurrences of each category in the 'Conflict Solved' column
category_counts = data['Conflict Solved'].value_counts()

# Print the counts
print("Counts for each category in the 'Conflict Solved' column:")
print(category_counts)
