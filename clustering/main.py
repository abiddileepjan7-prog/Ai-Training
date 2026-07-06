import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("scada_pipeline.csv")

# -------------------------------
# Drop unnecessary columns
# -------------------------------
df = df.drop(columns=["target", "event_type"], errors="ignore")

# -------------------------------
# Timestamp Feature Engineering
# -------------------------------
df["timestamp"] = pd.to_datetime(df["timestamp"])

df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month

df = df.drop(columns=["timestamp"])

# -------------------------------
# Remove ID column
# -------------------------------
df = df.drop(columns=["segment_id"], errors="ignore")

# -------------------------------
# Feature Matrix
# -------------------------------
X = df.copy()

# -------------------------------
# Standardization
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================================
# ELBOW METHOD (Only for KMeans)
# =====================================================
wcss = []

for k in range(1, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)
    wcss.append(model.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# =====================================================
# Number of clusters
# =====================================================
k = 3

# =====================================================
# KMeans
# =====================================================
kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_scaled)

kmeans_score = silhouette_score(
    X_scaled,
    kmeans_labels
)

print("\n========== KMEANS ==========")

print("Silhouette Score:",
      round(kmeans_score,3))

print("\nCluster Counts")

print(pd.Series(kmeans_labels).value_counts())

# =====================================================
# Agglomerative Clustering
# =====================================================
agg = AgglomerativeClustering(
    n_clusters=k
)

agg_labels = agg.fit_predict(X_scaled)

agg_score = silhouette_score(
    X_scaled,
    agg_labels
)

print("\n========== AGGLOMERATIVE ==========")

print("Silhouette Score:",
      round(agg_score,3))

print("\nCluster Counts")

print(pd.Series(agg_labels).value_counts())

# =====================================================
# Comparison Table
# =====================================================
comparison = pd.DataFrame({

    "Algorithm":[
        "KMeans",
        "Agglomerative"
    ],

    "Silhouette Score":[
        round(kmeans_score,3),
        round(agg_score,3)
    ]

})

print("\n==============================")
print("Algorithm Comparison")
print("==============================")

print(comparison)

best = comparison.loc[
    comparison["Silhouette Score"].idxmax()
]

print("\nBest Algorithm:", best["Algorithm"])

# =====================================================
# PCA
# =====================================================
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)







import numpy as np

# -------------------------------
# Comparison Histogram (Bar Plot)
# -------------------------------
algorithms = ["KMeans", "Agglomerative"]
scores = [kmeans_score, agg_score]

plt.figure(figsize=(7,5))

bars = plt.bar(algorithms, scores)

plt.title("Clustering Algorithm Comparison (Silhouette Score)")
plt.xlabel("Algorithm")
plt.ylabel("Silhouette Score")

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom"
    )

plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.show()









# --------------------------
# KMeans Plot
# --------------------------
plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=kmeans_labels,
    cmap="viridis"
)

plt.title("KMeans Clustering")

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.colorbar(label="Cluster")

plt.show()

# --------------------------
# Agglomerative Plot
# --------------------------
plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=agg_labels,
    cmap="plasma"
)

plt.title("Agglomerative Clustering")

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.colorbar(label="Cluster")

plt.show()