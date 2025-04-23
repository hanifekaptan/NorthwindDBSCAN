# Problem 3: Tedarikçi Segmentasyonu
# Veritabanı tabloları: Suppliers, Products, OrderDetails, Orders
# Amaç: Tedarikçileri sağladıkları ürünlerin satış performansına göre gruplandırmak.

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
# - Tedarik ettiği ürün sayısı (num_products_supplied)
# - Bu ürünlerin toplam satış miktarı (total_sales_quantity)
# - Ortalama satış fiyatı (avg_selling_price - using avg unit price from order_details for their products)
# - Toplam farklı müşteri sayısı (total_distinct_customers)
query = """
WITH SupplierStats AS (
    SELECT
        s.supplier_id,
        s.company_name,
        COUNT(DISTINCT p.product_id) AS num_products_supplied,
        SUM(od.quantity) AS total_sales_quantity,
        AVG(od.unit_price) AS avg_selling_price,
        COUNT(DISTINCT o.customer_id) AS total_distinct_customers
    FROM suppliers s
    JOIN products p ON s.supplier_id = p.supplier_id
    LEFT JOIN order_details od ON p.product_id = od.product_id
    LEFT JOIN orders o ON od.order_id = o.order_id
    WHERE od.order_id IS NOT NULL -- Ensure the supplier's products have been ordered
    GROUP BY s.supplier_id, s.company_name
    HAVING SUM(od.quantity) > 0 -- Ensure there are sales
)
SELECT * FROM SupplierStats;
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
features = ["num_products_supplied", "total_sales_quantity", "avg_selling_price", "total_distinct_customers"]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Optimal Epsilon (eps) Calculation ---
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
        optimal_eps = distances[kneedle.elbow] if kneedle.elbow else distances[-1] * 0.5 # Fallback

        # Plotting the distances
        plt.figure(figsize=(10, 6))
        plt.plot(distances, label='K-distance curve')
        if kneedle.elbow:
            plt.axvline(x=kneedle.elbow, color='r', linestyle='--', label=f'Optimal eps (Elbow): {optimal_eps:.3f}')
        else:
            plt.axhline(y=optimal_eps, color='orange', linestyle='--', label=f'Optimal eps (Fallback): {optimal_eps:.3f}')
        plt.xlabel('Suppliers sorted by distance to k-th neighbor')
        plt.ylabel(f'{min_samples}-th Nearest Neighbor Distance')
        plt.title('Elbow Method for Optimal eps in Supplier Segmentation')
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"Error finding optimal eps with kneed: {e}")
        mean_distance = np.mean(distances)
        optimal_eps = mean_distance
        print(f"Using fallback optimal eps (mean distance): {optimal_eps:.3f}")

        plt.figure(figsize=(10, 6))
        plt.plot(distances, label='K-distance curve')
        plt.axhline(y=optimal_eps, color='orange', linestyle='--', label=f'Optimal eps (Mean Fallback): {optimal_eps:.3f}')
        plt.xlabel('Suppliers sorted by distance to k-th neighbor')
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
# Visualize using num_products_supplied vs total_sales_quantity
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['num_products_supplied'], df['total_sales_quantity'], c=df['cluster'], cmap='viridis', s=60, alpha=0.7)
plt.xlabel("Tedarik Edilen Ürün Sayısı")
plt.ylabel("Toplam Satış Miktarı")
plt.title("Tedarikçi Segmentasyonu (DBSCAN) - Ürün Sayısı vs Satış Miktarı")
plt.grid(True)
plt.colorbar(scatter, label='Küme No (-1 = Aykırı)')
plt.show()

# Visualize using total_distinct_customers vs avg_selling_price
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['total_distinct_customers'], df['avg_selling_price'], c=df['cluster'], cmap='plasma', s=60, alpha=0.7)
plt.xlabel("Toplam Farklı Müşteri Sayısı")
plt.ylabel("Ortalama Satış Fiyatı (Ürünlerin)")
plt.title("Tedarikçi Segmentasyonu (DBSCAN) - Müşteri Sayısı vs Ortalama Fiyat")
plt.grid(True)
plt.colorbar(scatter, label='Küme No (-1 = Aykırı)')
plt.show()

# --- Outlier Analysis ---
outliers = df[df["cluster"] == -1]
print(f"Toplam tedarikçi sayısı: {len(df)}")
print(f"Küme sayısı (aykırılar hariç): {df[df['cluster'] != -1]['cluster'].nunique()}")
print(f"Aykırı (-1) olarak işaretlenen tedarikçi sayısı: {len(outliers)}")

if not outliers.empty:
    print("Aykırı Tedarikçiler (Örnek):")
    # Display potentially interesting columns for outliers
    print(outliers[['company_name'] + features].head())
else:
    print("Aykırı tedarikçi bulunamadı.")

print("Supplier segmentation analysis complete.")
# Save model and scaler
joblib.dump({"scaler": scaler, "dbscan": dbscan}, "model3.pkl")
print("Model saved to model3.pkl") 