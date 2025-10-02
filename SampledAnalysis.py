import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv')

# ---- UPDATED COLUMN SELECTION ----
columns_to_keep = [
    'Flow Duration',
    'Total Fwd Packet',
    'Total Bwd packets',
    'Fwd Packet Length Max',
    'Bwd Packet Length Max',
    'Fwd Packet Length Mean',
    'Bwd Packet Length Mean',
    'Flow IAT Mean',
    'Label',
    'Traffic Type'
]
# Keep only selected columns if needed
df_reduced = df[columns_to_keep].copy()

# For example, keep 1% of the data
sampled_df, _ = train_test_split(
    df_reduced,
    test_size=0.01,
    stratify=df_reduced['Label'],
    random_state=42
)

# Save to new CSV for further analysis
sampled_df.to_csv('data_sampled.csv', index=False)

print(f"Sampled dataset shape: {sampled_df.shape}")
print("Class distribution in the sampled dataset:")
print(sampled_df['Label'].value_counts(normalize=True))