import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("data.csv")

for col in df.columns:
    print(f"\n=== {col} ===")
    print("Type:", df[col].dtype)
    print(df[col].describe(include='all'))

    # Numeric columns
    if pd.api.types.is_numeric_dtype(df[col]):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        df[col].hist(bins=30)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)

        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.tight_layout()
        plt.show()

    # Categorical columns (object type with few unique)
    elif df[col].dtype == 'object' and df[col].nunique() < 30:
        plt.figure(figsize=(8, 4))
        df[col].value_counts().plot(kind='bar')
        plt.title(f'Value Counts of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()
    # Skip plots for columns that are long unique (e.g. IDs, IPs, Timestamps)

    elif df[col].nunique() > 30:
        print(f"Column {col} has {df[col].nunique()} unique values, skipping plot.")

# For correlation between numeric columns
num_cols = df.select_dtypes(include='number').columns
if len(num_cols) > 1:
    plt.figure(figsize=(16, 10))
    sns.heatmap(df[num_cols].corr(), annot=False, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()