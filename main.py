import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("Mall_Customers.csv")
df = df[["Age", "Annual Income (k$)",
         "Spending Score (1-100)"]]  # Только нужные признаки

scaled = StandardScaler().fit_transform(df)


def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def k_medoids(X, k, max_iter=200):
    m = X.shape[0]
    medoids_idx = np.random.choice(m, k, replace=False)
    medoids = X[medoids_idx]

    for _ in range(max_iter):
        labels = np.argmin(
            [[euclidean(x, medoid) for medoid in medoids] for x in X],
            axis=1)
        new_medoids = []
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) == 0:
                new_medoids.append(medoids[i])
                continue
            costs = [
                np.sum([euclidean(p, other) for other in cluster_points])
                for p in cluster_points]
            new_medoids.append(cluster_points[np.argmin(costs)])
        new_medoids = np.array(new_medoids)
        if np.all(medoids == new_medoids):
            break
        medoids = new_medoids

    return labels, medoids


def clara(X, k, n_samples=200, n_iter=15):
    best_labels = None
    best_cost = float('inf')
    best_medoids = None
    X = np.array(X)

    for _ in range(n_iter):
        sample_idx = np.random.choice(len(X), size=n_samples, replace=False)
        sample = X[sample_idx]
        labels, medoids = k_medoids(sample, k)

        full_labels = np.argmin(
            [[euclidean(x, medoid) for medoid in medoids] for x in X],
            axis=1)
        cost = sum([euclidean(x, medoids[full_labels[i]]) for i, x in
                    enumerate(X)])

        if cost < best_cost:
            best_cost = cost
            best_labels = full_labels
            best_medoids = medoids
    return best_labels, best_medoids


def dbscan(X, eps=0.9, min_pts=7):
    X = np.array(X)
    n = len(X)
    labels = np.full(n, -1)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    def region_query(p):
        return [i for i in range(n) if euclidean(X[p], X[i]) <= eps]

    def expand_cluster(p, neighbors):
        nonlocal cluster_id
        labels[p] = cluster_id
        i = 0
        while i < len(neighbors):
            n_point = neighbors[i]
            if not visited[n_point]:
                visited[n_point] = True
                new_neighbors = region_query(n_point)
                if len(new_neighbors) >= min_pts:
                    neighbors += new_neighbors
            if labels[n_point] == -1:
                labels[n_point] = cluster_id
            i += 1

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = region_query(i)
        if len(neighbors) < min_pts:
            labels[i] = -1
        else:
            expand_cluster(i, neighbors)
            cluster_id += 1

    return labels


clara_labels, clara_medoids = clara(scaled, k=5, n_iter=17, n_samples=150)
df["CLARA"] = clara_labels

dbscan_labels = dbscan(scaled, eps=0.4, min_pts=4)
df["DBSCAN"] = dbscan_labels

dbscan_centroids = []
for label in sorted(set(dbscan_labels)):
    if label == -1:
        continue  # Пропускаем шум
    cluster_points = scaled[np.array(dbscan_labels) == label]
    centroid = np.mean(cluster_points, axis=0)
    dbscan_centroids.append(centroid)
dbscan_centroids = np.array(dbscan_centroids)

scaler = StandardScaler().fit(
    df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]])
medoids_unscaled = scaler.inverse_transform(clara_medoids)
dbscan_centroids_unscaled = scaler.inverse_transform(dbscan_centroids)

fig = plt.figure(figsize=(14, 6))

ax = fig.add_subplot(121, projection='3d')
ax.scatter(df["Age"], df["Annual Income (k$)"],
           df["Spending Score (1-100)"],
           c=df["CLARA"], cmap="Set2", s=40)
ax.scatter(medoids_unscaled[:, 0], medoids_unscaled[:, 1],
           medoids_unscaled[:, 2],
           c='black', s=120, marker='X', label='CLARA Medoids')
ax.set_title("CLARA 3D")
ax.set_xlabel("Age")
ax.set_ylabel("Income")
ax.set_zlabel("Score")
ax.legend()

ax2 = fig.add_subplot(122, projection='3d')

mask = df["DBSCAN"] != -1
filtered_df = df[mask]
filtered_labels = df["DBSCAN"][mask]

ax2.scatter(filtered_df["Age"], filtered_df["Annual Income (k$)"],
            filtered_df["Spending Score (1-100)"],
            c=filtered_labels, cmap="Set2", s=40)

ax2.scatter(dbscan_centroids_unscaled[:, 0],
            dbscan_centroids_unscaled[:, 1],
            dbscan_centroids_unscaled[:, 2],
            c='black', s=120, marker='P', label='DBSCAN Centroids')

ax2.set_title("DBSCAN 3D (без шума)")
ax2.set_xlabel("Age")
ax2.set_ylabel("Income")
ax2.set_zlabel("Score")
ax2.legend()

plt.tight_layout()
plt.show()
