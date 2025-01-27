import pandas as pd

def load_and_process_data(file1, file2):
    # Load the CSV files
    label1_df = pd.read_csv(file1)
    label2_df = pd.read_csv(file2)
    
    # Correct any known typos in labels for both labelers
    corrections = {
        'Simantically Equivalent': 'Semantically Equivalent',  # common typo correction
        'Partally Equivalent': 'Partially Equivalent',  # additional typo if found
        'Meangiful Code Description': 'Meaningful Code Description'  # another common typo
    }
    
    label1_df['Label'] = label1_df['Label'].replace(corrections)
    label2_df['Label'] = label2_df['Label'].replace(corrections)
    
    # Create a cross-tabulation of the labels
    cross_tab = pd.crosstab(label1_df['Label'], label2_df['Label'])
    
    return cross_tab

def main():
    file1 = 'path/to/your/first/label/file.csv'  # Update with actual file path
    file2 = 'path/to/your/second/label/file.csv'  # Update with actual file path
    result_matrix = load_and_process_data(file1, file2)
    
    # Print the resulting cross-tabulation matrix
    print("Cross-tabulation of labels:")
    print(result_matrix)

if __name__ == "__main__":
    main()
