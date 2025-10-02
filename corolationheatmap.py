import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("data.csv")

# Select only numeric columns
num_cols = df.select_dtypes(include='number')

# Create correlation heatmap if more than one numeric column exists
if num_cols.shape[1] > 1:
    plt.figure(figsize=(20, 12))  # Increased size
    sns.heatmap(num_cols.corr(), annot=False, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()
else:
    print("Not enough numeric columns for a correlation heatmap.")