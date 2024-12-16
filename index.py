import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
try:
    # Load the Iris dataset
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    # Display first few rows
    print("First few rows of the dataset:")
    print(data.head())

    # Check dataset structure
    print("\nDataset Information:")
    print(data.info())

    # Check for missing values
    print("\nMissing Values:")
    print(data.isnull().sum())

    # No missing values in the Iris dataset, but if there were:
    # data = data.dropna()  # Dropping rows with missing values
    # or data.fillna(value, inplace=True)  # Filling missing values
except FileNotFoundError:
    print("Error: The specified dataset file was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# Task 2: Basic Data Analysis
try:
    # Compute basic statistics
    print("\nBasic Statistics:")
    print(data.describe())

    # Grouping by a categorical column ('species') and computing the mean of numerical columns
    group_column = 'species'
    grouped_data = data.groupby(group_column).mean()
    print("\nMean values by species:")
    print(grouped_data)

    # Identify patterns or findings
    print("\nFindings:")
    print("- Setosa species has the smallest average sepal and petal sizes.")
    print("- Virginica species has the largest average petal width and length.")

except Exception as e:
    print(f"An error occurred during analysis: {e}")

# Task 3: Data Visualization
try:
    # Line Chart: Illustrative example (not time-series data)
    plt.figure(figsize=(10, 6))
    for species, group in data.groupby('species'):
        plt.plot(group.index, group['sepal length (cm)'], label=f'{species} Sepal Length')
    plt.title('Line Chart: Sepal Length by Index')
    plt.xlabel('Index')
    plt.ylabel('Sepal Length (cm)')
    plt.legend()
    plt.show()

    # Bar Chart: Average petal length per species
    plt.figure(figsize=(10, 6))
    grouped_data['petal length (cm)'].plot(kind='bar', color='skyblue', alpha=0.8)
    plt.title('Bar Chart: Average Petal Length by Species')
    plt.xlabel('Species')
    plt.ylabel('Petal Length (cm)')
    plt.show()

    # Histogram: Distribution of petal width
    plt.figure(figsize=(10, 6))
    sns.histplot(data['petal width (cm)'], kde=True, color='purple')
    plt.title('Histogram: Distribution of Petal Width')
    plt.xlabel('Petal Width (cm)')
    plt.ylabel('Frequency')
    plt.show()

    # Scatter Plot: Sepal length vs petal length
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data['sepal length (cm)'], y=data['petal length (cm)'], hue=data['species'], palette='viridis')
    plt.title('Scatter Plot: Sepal Length vs Petal Length by Species')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.legend(title='Species')
    plt.show()

except Exception as e:
    print(f"An error occurred during visualization: {e}")
