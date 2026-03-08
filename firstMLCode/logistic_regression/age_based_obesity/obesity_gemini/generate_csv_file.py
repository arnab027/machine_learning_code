import pandas as pd
import numpy as np

def expand_health_csv(filename='D://pythonMachineLearning//firstMLCode//logistic_regression//age_based_obesity//obesity_gemini//obesity_geminihealth_data.csv', num_records=500):
    # 1. Generate realistic BMI distribution (15 to 50)
    # We use a normal distribution centered around 28 to get more 'borderline' cases
    new_bmis = np.random.normal(28, 7, num_records)
    new_bmis = np.clip(new_bmis, 15, 55) # Keep within human limits
    
    # 2. Assign labels based on the BMI 30 threshold
    # We add 'probabilistic noise' so the model learns a curve, not a hard cliff
    # This prevents the Maximum Likelihood engine from hitting numerical errors
    noise = np.random.normal(0, 1.2, num_records)
    new_labels = ((new_bmis + noise) >= 30).astype(int)
    
    new_data = pd.DataFrame({
        'BMI': np.round(new_bmis, 2),
        'Is_Obese': new_labels
    })
    
    # 3. Append to existing file or create new one
    try:
        new_data.to_csv(filename, mode='a', header=False, index=False)
        print(f"Successfully added {num_records} records to {filename}.")
    except FileNotFoundError:
        new_data.to_csv(filename, index=False)
        print(f"Created new file {filename} with {num_records} records.")

expand_health_csv()