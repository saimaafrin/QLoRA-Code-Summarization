import pandas as pd

# Load the original data file
file_path = 'projects/QLoRA/results/csv_files/statistical_analysis/Experiments-Tracker - Conflict_solved_CodeLlama-34B-qlora-java.csv'
data = pd.read_csv(file_path)

# Filter the data to include only the columns for "Labeller-1" and "Labeller-2"
filtered_data = data[["Label-1", "Label-2"]]

# Define a mapping from textual rating categories to numerical categories
category_map = {
    "Semantically Equivalent": 1,
    "Partially Equivalent": 2,
    "Meaningful Code Description": 3,
    "Incorrect": 4  
}

# Replace textual categories with their corresponding numerical values
filtered_data = filtered_data.replace(category_map)
# Replace NaN values with a placeholder that pandas can keep as integer type
filtered_data = filtered_data.fillna(-1)  # Using -1 as a placeholder for missing values
# Convert columns to integer type
filtered_data = filtered_data.astype(int)
# Replace placeholder with 'NA' in the final output
filtered_data = filtered_data.replace(-1, 'NA')
filtered_data = filtered_data.head(384)

# Replace NaN values with 'NA' to denote missing values
filtered_data.fillna('NA', inplace=True)

# Save the resulting data to a CSV file without headers or index
output_path = 'projects/QLoRA/results/csv_files/statistical_analysis/K-Alpha_Calculator_Data_CL34_java.csv'
filtered_data.to_csv(output_path, index=False, header=False)

# Display the path to the saved file, the number of tasks, and the number of categories
print(f"File saved to: {output_path}")
print(f"Number of tasks: {filtered_data.shape[0]}")
print(f"Number of rating categories: {len(category_map)}")
