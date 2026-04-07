import pandas as pd

# dataset load (Excel file)
data = pd.read_excel("dataset/dataset_project.xlsx")

# first rows print
print("First 5 Rows:\n")
print(data.head())

# columns print
print("\nColumns:\n")
print(data.columns)

# dataset info
print("\nDataset Info:\n")
print(data.info())