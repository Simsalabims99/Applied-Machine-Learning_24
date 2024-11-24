import pandas as pd

# Setting the path to the dataset (update with correct path to dataset)
dataset_path = "/Users/simonpedersen/Downloads/Boligpriser - Uncleaned.csv"

# Reading in dataset
data = pd.read_csv(dataset_path)

# Dataset description
num_attributes = data.shape[1]
print(f"The dataset contains {num_attributes} attributes.\n")

print("Attributes and their data types:")
print(data.dtypes)
print("\n")

# Sorting the dataset based on sales date to establish time-range
time_range = data.sort_values(by='Salgsdato')
print("Sorted data by 'Sales date'")
print(time_range.head)

print("\ntime range of sales:")
print(f"From: {time_range['Salgsdato'].iloc[0]} To: {time_range['Salgsdato'].iloc[-1]}")

# Overview of columns with null values and their counts
print("\nMissing values per attribute:")
print(data.isnull().sum())

# Detalied description of all attributes
print("\nDataset Attribute Descriptions:\n")

for col in data.columns:
    print(f"Attribute: {col}")
    print(f"Data Type: {data[col].dtype}")
    print(f"Unique Values: {data[col].nunique()}")

    if data[col].dtype == 'object':  # Categorical attribute
        print("Sample Values:", data[col].value_counts().head())
    else:  # Numerical attribute
        print("Statistics:")
        print(data[col].describe().apply(lambda x: format(x, 'f')))
    
    print("\n" + "-"*30 + "\n")

