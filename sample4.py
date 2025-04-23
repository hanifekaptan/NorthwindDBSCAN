# Problem 4: Ülkelere Göre Satış Deseni Analizi
# Veritabanı tabloları: Customers, Orders, OrderDetails
# Amaç: Farklı ülkelerden gelen siparişleri DBSCAN ile kümeleyip sıra dışı alışkanlıkları tespit etmek.

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
password = "12345" # Consider using environment variables or a config file for credentials
host = "localhost"
port = "5432"
database = "Northwind"

engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}")

# --- SQL Query ---
# Features:
# - Toplam sipariş (total_orders)
# - Ortalama sipariş tutarı (avg_order_value)
# - Sipariş başına ortalama ürün miktarı (avg_quantity_per_order)

query = """
WITH CountryOrderStats AS (
    SELECT
        c.country,
        COUNT(DISTINCT o.order_id) AS total_orders,
        SUM(od.unit_price * od.quantity * (1 - od.discount)) / COUNT(DISTINCT o.order_id) AS avg_order_value,
        SUM(od.quantity) / COUNT(DISTINCT o.order_id)::decimal AS avg_quantity_per_order -- Use decimal for precision
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN order_details od ON o.order_id = od.order_id
    WHERE c.country IS NOT NULL
    GROUP BY c.country
    HAVING COUNT(DISTINCT o.order_id) > 1 -- Consider countries with more than 1 order for meaningful analysis
)
SELECT * FROM CountryOrderStats;
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
    print("No data retrieved from the database (check query or data source). Exiting.")
    exit()

# --- Feature Selection and Scaling ---
features = ["total_orders", "avg_order_value", "avg_quantity_per_order"]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Optimal Epsilon (eps) Calculation ---
n_features = X_scaled.shape[1]
min_samples = max(n_features + 1, 5) # Heuristic: n_features + 1, but min 5 for stability
# Using max(n+1, 5) as n=3 is small

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
        plt.xlabel('Countries sorted by distance to k-th neighbor')
        plt.ylabel(f'{min_samples}-th Nearest Neighbor Distance')
        plt.title('Elbow Method for Optimal eps in Country Sales Pattern Analysis')
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
        plt.xlabel('Countries sorted by distance to k-th neighbor')
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
# Visualize using total_orders vs avg_order_value
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['total_orders'], df['avg_order_value'], c=df['cluster'], cmap='viridis', s=60, alpha=0.7)
plt.xlabel("Toplam Sipariş Sayısı")
plt.ylabel("Ortalama Sipariş Tutarı")
plt.title("Ülke Bazında Satış Deseni (DBSCAN) - Sipariş Sayısı vs Ortalama Tutar")
plt.grid(True)
plt.colorbar(scatter, label='Küme No (-1 = Aykırı)')

# Optional: Add labels for outliers or interesting points
# for i, txt in enumerate(df['country']):
#     if df['cluster'].iloc[i] == -1: # Label outliers
#         plt.annotate(txt, (df['total_orders'].iloc[i], df['avg_order_value'].iloc[i]))

plt.show()

# Visualize using avg_quantity_per_order vs avg_order_value
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['avg_quantity_per_order'], df['avg_order_value'], c=df['cluster'], cmap='plasma', s=60, alpha=0.7)
plt.xlabel("Sipariş Başına Ortalama Ürün Miktarı")
plt.ylabel("Ortalama Sipariş Tutarı")
plt.title("Ülke Bazında Satış Deseni (DBSCAN) - Ürün Miktarı vs Ortalama Tutar")
plt.grid(True)
plt.colorbar(scatter, label='Küme No (-1 = Aykırı)')
plt.show()

# --- Outlier Analysis ---
outliers = df[df["cluster"] == -1]
print(f"Toplam ülke sayısı (analiz edilen): {len(df)}")
print(f"Küme sayısı (aykırılar hariç): {df[df['cluster'] != -1]['cluster'].nunique()}")
print(f"Aykırı (-1) olarak işaretlenen ülke sayısı: {len(outliers)}")

if not outliers.empty:
    print("Aykırı Ülkeler (Sıra Dışı Satış Deseni):")
    # Display potentially interesting columns for outliers
    print(outliers[['country'] + features])
else:
    print("Aykırı ülke bulunamadı.")

print("Country sales pattern analysis complete.")
# Save model and scaler
joblib.dump({"scaler": scaler, "dbscan": dbscan}, "model4.pkl")
print("Model saved to model4.pkl") 
