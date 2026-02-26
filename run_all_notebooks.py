import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

notebooks = [
    '01_EDA.ipynb',
    '02_Preprocessing.ipynb',
    '03_Binary_Classification.ipynb',
    '04_Family_Classification.ipynb',
    '05_SubType_Classification.ipynb',
    '06_Robustness.ipynb',
    '07_Explainability.ipynb',
    '08_Final_Report.ipynb',
    'train_full_model.ipynb'
]

ep = ExecutePreprocessor(timeout=1800, kernel_name='python3')

for nb_file in notebooks:
    if not os.path.exists(nb_file):
        print(f"Skipping {nb_file} - doesn't exist.")
        continue
        
    print(f"Executing {nb_file}...")
    try:
        with open(nb_file, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        ep.preprocess(nb, {'metadata': {'path': './'}})
        with open(nb_file, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"Success: {nb_file}")
    except Exception as e:
        print(f"Failed {nb_file}: {str(e).encode('utf-8', 'ignore').decode('utf-8')}")
