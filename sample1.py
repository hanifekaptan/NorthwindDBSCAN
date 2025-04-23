#Müşterilerin alışveriş davranışlarına göre gruplanması ve aykırı verilerin keşfi

#order_details,customers,orders

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
import joblib

user = "postgres"
password = "12345"
host = "localhost"
port = "5432"
database = "Northwind"

engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}")

query = """
select 
c.customer_id,
count(o.order_id) as total_orders,
sum(od.unit_price*od.quantity) as total_spent,
avg(od.unit_price*od.quantity) as avg_order_value
from customers c inner join orders o
on c.customer_id =o.customer_id
inner join order_details od
on o.order_id = od.order_id
group by c.customer_id
having count(o.order_id)>0
"""

df = pd.read_sql_query(query,engine)
print(df.head())

X = df[["total_orders","total_spent","avg_order_value"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def find_optimal_eps(X_scaled,min_samples=3):
    neighbors = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
    distances,_  = neighbors.kneighbors(X_scaled)

    distances = np.sort(distances[:,min_samples-1])

    kneedle = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
    optimal_eps = distances[kneedle.elbow]

    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.axvline(x=kneedle.elbow, color='r', linestyle='--', label=f'Optimal eps: {optimal_eps:.2f}')
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{min_samples}-th nearest neighbor distance')
    plt.title('Elbow Method for Optimal eps')
    plt.legend()
    plt.grid(True)
    plt.show()

    return optimal_eps

# --- Min_samples Optimization ---
def optimize_min_samples(X_scaled, ms_values):
    best_ms, best_eps, best_score = None, None, -1
    for ms in ms_values:
        # Find eps for this min_samples
        eps = find_optimal_eps(X_scaled, ms)
        labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(X_scaled)
        unique_labels = set(labels)
        # Require at least 2 clusters for silhouette_score
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
ms_values = list(range(2, 2 * X_scaled.shape[1] + 2))  # e.g., 2 to 2*n_features+1
best_ms, best_eps = optimize_min_samples(X_scaled, ms_values)
dbscan = DBSCAN(eps=best_eps, min_samples=best_ms)

df["cluster"] = dbscan.fit_predict(X_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(df['total_orders'], df['total_spent'], c=df['cluster'], cmap='plasma', s=60)
plt.xlabel("Toplam Sipariş Sayısı")
plt.ylabel("Toplam Harcama")
plt.title("Müşteri Segmentasyonu (DBSCAN)")
plt.grid(True)
plt.colorbar(label='Küme No')
plt.show()

outliers = df[df["cluster"]==-1]
print("Aykırı veri sayısı : ", len(outliers))
print(outliers[["customer_id","total_orders","total_spent"]])
# Save model and scaler
joblib.dump({"scaler": scaler, "dbscan": dbscan}, "model1.pkl")
print("Model saved to model1.pkl")
