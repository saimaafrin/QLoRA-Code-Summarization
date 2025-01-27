import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests


def calculate_tests(data, column1, column2):
    # Build the contingency table
    yes_yes = data[(data[column1] == 1) & (data[column2] == 1)].shape[0]
    yes_no = data[(data[column1] == 1) & (data[column2] == 0)].shape[0]
    no_yes = data[(data[column1] == 0) & (data[column2] == 1)].shape[0]
    no_no = data[(data[column1] == 0) & (data[column2] == 0)].shape[0]

    # Adjusting the counts by adding small value in the contingency table to avoid zero values
    yes_yes += 1.0
    yes_no += 1.0
    no_yes += 1.0
    no_no += 1.0

    table = [[yes_yes, yes_no], [no_yes, no_no]]
    print(table)

    # Perform McNemar's test
    mcnemar_result = mcnemar(table, exact=False, correction=True)
    p_value = mcnemar_result.pvalue
    
    # Calculate Odds Ratio
    #odds_ratio = 'Undefined' if yes_no == 0 or no_yes == 0 else yes_no / no_yes
    odds_ratio = (yes_no * no_yes) / (yes_yes * no_no)

    return p_value, odds_ratio

def main():
    data = pd.read_csv('projects/QLoRA/results/csv_files/statistical_analysis/McNemar-OR.csv')
    
    comparisons = [
        ('Is_perfect_DSC_1.3_full_python', 'Is_perfect_phi_3_mini_full_python'),
        ('Is_perfect_DSC_1.3_qlora_python', 'Is_perfect_phi_3_mini_qlora_python'),
        ('Is_perfect_DSC_1.3_full_python', 'Is_perfect_DSC_1.3_qlora_python'),
        ('Is_perfect_phi_3_mini_full_python', 'Is_perfect_phi_3_mini_qlora_python'),
        ('Is_perfect_DSC_1.3_full_java', 'Is_perfect_phi_3_mini_full_java'),
        ('Is_perfect_DSC_1.3_qlora_java', 'Is_perfect_phi_3_mini_qlora_java'),
        ('Is_perfect_DSC_1.3_full_java', 'Is_perfect_DSC_1.3_qlora_java'),
        ('Is_perfect_phi_3_mini_full_java', 'Is_perfect_phi_3_mini_qlora_java')
    ]

    '''results = {}
    for col1, col2 in comparisons:
        p_value, odds_ratio = calculate_tests(data, col1, col2)
        results[(col1, col2)] = (p_value, odds_ratio)
    
    for key, (p, oratio) in results.items():
        print(f"Comparison: {key[0]} vs {key[1]} - P-value: {p}, Odds Ratio: {oratio}")'''
    
    pvals = []
    results = {}
    for col1, col2 in comparisons:
        p_value, odds_ratio = calculate_tests(data, col1, col2)
        results[(col1, col2)] = (p_value, odds_ratio)
        pvals.append(p_value)
    
    # Applying Holm's Correction
    corrected_pvals = multipletests(pvals, alpha=0.05, method='holm')[1]

    # Output results
    for i, ((col1, col2), (original_pval, odds_ratio)) in enumerate(zip(comparisons, results.values())):
        print(f"Comparison: {col1} vs {col2} - Original P-value: {original_pval}, Adjusted P-value: {corrected_pvals[i]}, Odds Ratio: {odds_ratio}")


if __name__ == "__main__":
    main()
