import pandas as pd
from sklearn.cluster import MiniBatchKMeans  # Faster for large datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load data
df = pd.read_csv('data.csv')

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
df_reduced = df[columns_to_keep].copy()

clustering_features = [
    'Flow Duration',
    'Total Fwd Packet',
    'Total Bwd packets',
    'Fwd Packet Length Max',
    'Bwd Packet Length Max',
    'Fwd Packet Length Mean',
    'Bwd Packet Length Mean',
    'Flow IAT Mean'
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_reduced[clustering_features])

# --- Set number of clusters for ~10,000 output samples ---
n_clusters = min(10000, len(df_reduced))

# Use MiniBatchKMeans for speed on large data
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000, n_init=5)
labels = kmeans.fit_predict(X_scaled)
cluster_centers = kmeans.cluster_centers_

closest_indices = []
for i in range(n_clusters):
    cluster_points = np.where(labels == i)[0]
    if len(cluster_points) > 0:
        distances = np.linalg.norm(X_scaled[cluster_points] - cluster_centers[i], axis=1)
        closest = cluster_points[np.argmin(distances)]
        closest_indices.append(closest)

cluster_representatives = df_reduced.iloc[closest_indices]

# Guarantee all rare instances (e.g., 'Benign') are included
for rare_label in ['Benign']:  # Add more labels if needed
    rare_rows = df_reduced[df_reduced['Label'] == rare_label]
    cluster_representatives = pd.concat([cluster_representatives, rare_rows]).drop_duplicates()

cluster_representatives.to_csv('data_clustered.csv', index=False)

print(f"Clustered set shape: {cluster_representatives.shape}")
print("Class distribution in the clustered dataset:")
print(cluster_representatives['Label'].value_counts(normalize=True))