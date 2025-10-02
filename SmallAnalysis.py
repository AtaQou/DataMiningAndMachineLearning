import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Helper to safely create filenames
def sanitize_filename(col):
    return re.sub(r'[\\/:"*?<>| ]+', '_', col)

# Load data (update filename as needed)
df = pd.read_csv("data.csv")

# List your most important numeric columns (adjust as needed)
num_cols = [
    'Flow Duration',
    'Total Fwd Packet',
    'Total Bwd packets',
    'Total Length of Fwd Packet',
    'Total Length of Bwd Packet',
    'Fwd Packet Length Mean',
    'Bwd Packet Length Mean',
    'Flow Bytes/s',
    'Flow Packets/s'
]

# List your most important categorical columns
cat_cols = [
    'Label',
    'Traffic Type'
]

# Generate histograms and boxplots for important numeric columns
for col in num_cols:
    if col in df.columns:
        safe_col = sanitize_filename(col)
        plt.figure(figsize=(12, 4))

        # Histogram
        plt.subplot(1, 2, 1)
        sns.histplot(df[col].dropna(), bins=30, kde=True)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)

        # Boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col].dropna())
        plt.title(f'Boxplot of {col}')

        plt.tight_layout()
        plt.savefig(f"{safe_col}_hist_box.png")
        plt.close()

# Bar plots for important categorical columns
for col in cat_cols:
    if col in df.columns:
        safe_col = sanitize_filename(col)
        value_counts = df[col].value_counts()
        plt.figure(figsize=(8, 4))
        sns.barplot(x=value_counts.index, y=value_counts.values, palette="Set2")
        plt.title(f'Value Counts of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(f"{safe_col}_barplot.png")
        plt.close()

# Correlation heatmap only for selected numeric columns (if more than one)
present_num_cols = [col for col in num_cols if col in df.columns]
if len(present_num_cols) > 1:
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[present_num_cols].corr(), annot=False, cmap='coolwarm')
    plt.title("Correlation Heatmap (Selected Columns)")
    plt.tight_layout()
    plt.savefig("correlation_heatmap_selected.png")
    plt.close()

print("Selected important graphs saved for your report.")