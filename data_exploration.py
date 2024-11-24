import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Visual representations of the target variable

# 1. Distribution of Sale Prices
plt.figure(figsize=(10, 6))
sns.histplot(data['Salgspris'], bins=50, kde=True, color='blue')
plt.title('Distribution of Sale Prices', fontsize=16)
plt.xlabel('Sale Price (M DKK)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('sale_price_distribution.png')
plt.show()

# Sale Prices by Region
region_prices = data.groupby('Region')['Salgspris'].median().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
region_prices.plot(kind='bar', color='orange')
plt.title('Median Sale Price by Region', fontsize=16)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Median Sale Price (M DKK)', fontsize=12)
plt.tight_layout()
plt.savefig('median_sale_price_by_region.png')
plt.show()

# Sale Prices by Property Type
property_prices = data.groupby('Boligtype')['Salgspris'].median().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
property_prices.plot(kind='bar', color='green')
plt.title('Median Sale Price by Residence Type', fontsize=16)
plt.xlabel('Property Type', fontsize=12)
plt.ylabel('Median Sale Price (M DKK)', fontsize=12)
plt.tight_layout()
plt.savefig('median_sale_price_by_property_type.png')
plt.show()