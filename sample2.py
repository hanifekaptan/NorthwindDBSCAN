# Problem 2: Ürün Kümeleme (Benzer Ürünler)
# Veritabanı tabloları: Products, OrderDetails, Orders
# Amaç: Benzer sipariş geçmişine sahip ürünleri DBSCAN ile gruplandırmak.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import joblib
from sklearn.metrics import silhouette_score

# --- Database Connection ---
# Reuse credentials from sample1.py
user = "postgres"
password = "HFN8874k." # Consider using environment variables or a config file for credentials
host = "localhost"
port = "5432"
database = "Northwind"

engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}")

# --- SQL Query ---
# Features:
# - Ortalama satış fiyatı (avg_price)
# - Satış sıklığı (sales_frequency: number of distinct orders)
# - Sipariş başına ortalama miktar (avg_quantity_per_order)
# - Kaç farklı müşteriye satıldı (distinct_customers)
query = """
WITH ProductStats AS (
    SELECT
        p.product_id,
        p.product_name,
        AVG(od.unit_price) AS avg_price,
        COUNT(DISTINCT o.order_id) AS sales_frequency,
        AVG(od.quantity) AS avg_quantity_per_order,
        COUNT(DISTINCT o.customer_id) AS distinct_customers
    FROM products p
    JOIN order_details od ON p.product_id = od.product_id
    JOIN orders o ON od.order_id = o.order_id
    GROUP BY p.product_id, p.product_name
    HAVING COUNT(DISTINCT o.order_id) > 0 -- Ensure product has been ordered
)
SELECT * FROM ProductStats;
"""

# --- Data Loading ---
try:
    df = pd.read_sql_query(query, engine)
    print("Data loaded successfully:")
    print(df.head())
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

if df.empty:
    print("No data retrieved from the database. Exiting.")
    exit()

# --- Feature Selection and Scaling ---
features = ["avg_price", "sales_frequency", "avg_quantity_per_order", "distinct_customers"]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Optimal Epsilon (eps) Calculation ---
# Note: min_samples is a hyperparameter. 2 * n_features is a common heuristic.
# Further optimization (ARGE) might involve testing different values.
n_features = X_scaled.shape[1]
min_samples = 2 * n_features # Heuristic: 2 * number of features

def find_optimal_eps(X_scaled, min_samples):
    print(f"Calculating optimal eps using min_samples = {min_samples}")
    neighbors = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
    distances, _ = neighbors.kneighbors(X_scaled)

    # Sort distance to the k-th neighbor
    distances = np.sort(distances[:, min_samples - 1])

    # Use KneeLocator to find the elbow point
    try:
        kneedle = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
        optimal_eps = distances[kneedle.elbow] if kneedle.elbow else distances[-1] * 0.5 # Fallback if no elbow detected

        # Plotting the distances
        plt.figure(figsize=(10, 6))
        plt.plot(distances, label='K-distance curve')
        if kneedle.elbow:
            plt.axvline(x=kneedle.elbow, color='r', linestyle='--', label=f'Optimal eps (Elbow): {optimal_eps:.3f}')
        else:
             plt.axhline(y=optimal_eps, color='orange', linestyle='--', label=f'Optimal eps (Fallback): {optimal_eps:.3f}')
        plt.xlabel('Products sorted by distance to k-th neighbor')
        plt.ylabel(f'{min_samples}-th Nearest Neighbor Distance')
        plt.title('Elbow Method for Optimal eps in Product Clustering')
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"Error finding optimal eps with kneed: {e}")
        # Fallback: Calculate eps based on mean distance or a fixed percentile
        mean_distance = np.mean(distances)
        optimal_eps = mean_distance
        print(f"Using fallback optimal eps (mean distance): {optimal_eps:.3f}")

        plt.figure(figsize=(10, 6))
        plt.plot(distances, label='K-distance curve')
        plt.axhline(y=optimal_eps, color='orange', linestyle='--', label=f'Optimal eps (Mean Fallback): {optimal_eps:.3f}')
        plt.xlabel('Products sorted by distance to k-th neighbor')
        plt.ylabel(f'{min_samples}-th Nearest Neighbor Distance')
        plt.title('Elbow Method for Optimal eps (Fallback)')
        plt.legend()
        plt.grid(True)
        plt.show()


    return optimal_eps

# --- Min_samples Optimization ---
def optimize_min_samples(X_scaled, ms_values):
    best_ms, best_eps, best_score = None, None, -1
    for ms in ms_values:
        eps = find_optimal_eps(X_scaled, ms)
        labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(X_scaled)
        unique_labels = set(labels)
        if len(unique_labels) > 1 and len(unique_labels) < len(X_scaled):
            try:
                score = silhouette_score(X_scaled, labels)
            except Exception:
                score = -1
            if score > best_score:
                best_score, best_ms, best_eps = score, ms, eps
    print(f"Optimized min_samples: {best_ms}, silhouette score: {best_score:.3f}, eps: {best_eps:.3f}")
    return best_ms, best_eps

# Optimize min_samples and eps using ARGE
ms_values = list(range(2, 2 * X_scaled.shape[1] + 2))
best_ms, best_eps = optimize_min_samples(X_scaled, ms_values)
print(f"Using eps={best_eps:.3f} and min_samples={best_ms}")

# --- DBSCAN Clustering with optimized parameters ---
dbscan = DBSCAN(eps=best_eps, min_samples=best_ms)
df["cluster"] = dbscan.fit_predict(X_scaled)

# --- Visualization ---
# Visualize using two prominent features, e.g., sales_frequency vs avg_price
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['sales_frequency'], df['avg_price'], c=df['cluster'], cmap='viridis', s=60, alpha=0.7)
plt.xlabel("Satış Sıklığı (Sipariş Sayısı)")
plt.ylabel("Ortalama Satış Fiyatı")
plt.title("Ürün Kümeleme (DBSCAN) - Satış Sıklığı vs Ortalama Fiyat")
plt.grid(True)
plt.colorbar(scatter, label='Küme No (-1 = Aykırı)')
plt.show()

# Another visualization: distinct_customers vs avg_quantity_per_order
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['distinct_customers'], df['avg_quantity_per_order'], c=df['cluster'], cmap='plasma', s=60, alpha=0.7)
plt.xlabel("Farklı Müşteri Sayısı")
plt.ylabel("Sipariş Başına Ortalama Miktar")
plt.title("Ürün Kümeleme (DBSCAN) - Müşteri Sayısı vs Ortalama Miktar")
plt.grid(True)
plt.colorbar(scatter, label='Küme No (-1 = Aykırı)')
plt.show()

# --- Outlier Analysis ---
outliers = df[df["cluster"] == -1]
print(f"Toplam ürün sayısı: {len(df)}")
print(f"Küme sayısı (aykırılar hariç): {df[df['cluster'] != -1]['cluster'].nunique()}")
print(f"Aykırı (-1) olarak işaretlenen ürün sayısı: {len(outliers)}")

if not outliers.empty:
    print("Aykırı Ürünler (Örnek):")
    # Display potentially interesting columns for outliers
    print(outliers[['product_name'] + features].head())
else:
    print("Aykırı ürün bulunamadı.")

print("Product clustering analysis complete.")
# Save model and scaler
joblib.dump({"scaler": scaler, "dbscan": dbscan}, "model2.pkl")
print("Model saved to model2.pkl") 