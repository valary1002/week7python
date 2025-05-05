# iris_data_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
print("\n--- Task 1: Load and Explore the Dataset ---")
try:
    iris = load_iris(as_frame=True)
    df = iris.frame

    # Display first few rows
    print("\nFirst 5 rows of the dataset:")
    print(df.head())

    # Check data types and missing values
    print("\nData types and non-null values:")
    print(df.info())

    print("\nMissing values in each column:")
    print(df.isnull().sum())

except FileNotFoundError:
    print("File not found. Please check the file name or path.")
except Exception as e:
    print(f"An error occurred: {e}")

# Task 2: Basic Data Analysis
print("\n--- Task 2: Basic Data Analysis ---")

# Basic statistics
print("\nDescriptive statistics:")
print(df.describe())

# Add species names
df['species'] = df['target'].map(dict(zip(range(3), iris.target_names)))

# Group by species and compute means
print("\nMean values grouped by species:")
print(df.groupby('species').mean())

# Task 3: Data Visualization
print("\n--- Task 3: Data Visualization ---")

# 1. Line chart (example with index as time)
df['index'] = df.index
plt.figure(figsize=(8, 4))
plt.plot(df['index'], df['sepal length (cm)'], label='Sepal Length')
plt.title('Line Chart of Sepal Length')
plt.xlabel('Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.tight_layout()
plt.show()

# 2. Bar chart
plt.figure(figsize=(6, 4))
sns.barplot(data=df, x='species', y='petal length (cm)', ci=None)
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.tight_layout()
plt.show()

# 3. Histogram
plt.figure(figsize=(6, 4))
plt.hist(df['sepal width (cm)'], bins=10, color='skyblue', edgecolor='black')
plt.title('Histogram of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 4. Scatter plot
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.tight_layout()
plt.show()

print("\nAnalysis complete. Visualizations displayed.")
