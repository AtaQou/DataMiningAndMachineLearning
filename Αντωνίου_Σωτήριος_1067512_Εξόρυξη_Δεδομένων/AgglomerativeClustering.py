import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import numpy as np

# Load the data (adjust the filename as needed)
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

# Subsample to avoid memory errors
SUBSAMPLE_SIZE = 2000   # Try 1000 or 2000; increase only if you don't hit OOM issues
df_sampled = df_reduced.sample(n=SUBSAMPLE_SIZE, random_state=42)

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
X_scaled = scaler.fit_transform(df_sampled[clustering_features])

n_clusters = 100   # Adjust as needed; should be << SUBSAMPLE_SIZE

agg = AgglomerativeClustering(n_clusters=n_clusters)
labels = agg.fit_predict(X_scaled)

# For each cluster, pick the point closest to the cluster's mean
closest_indices = []
for cluster_id in range(n_clusters):
    cluster_points = np.where(labels == cluster_id)[0]
    mean_vec = X_scaled[cluster_points].mean(axis=0)
    distances = np.linalg.norm(X_scaled[cluster_points] - mean_vec, axis=1)
    closest = cluster_points[np.argmin(distances)]
    closest_indices.append(closest)

cluster_representatives = df_sampled.iloc[closest_indices]
cluster_representatives.to_csv('data_clustered_agglomerative.csv', index=False)

print(f"Agglomerative clustered set shape: {cluster_representatives.shape}")
print("Class distribution in the agglomerative clustered dataset:")
print(cluster_representatives['Label'].value_counts(normalize=True))