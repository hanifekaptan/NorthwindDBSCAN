from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
from sklearn.cluster import DBSCAN

class ProductData(BaseModel):
    avg_price: float
    sales_frequency: int
    avg_quantity_per_order: float
    distinct_customers: int

    class Config:
        json_schema_extra = {
            "example":
                {
                    "avg_price": 25.50,
                    "sales_frequency": 150,
                    "avg_quantity_per_order": 5.2,
                    "distinct_customers": 85
                }
        }

app = FastAPI(title="Product Clustering API")

# Load trained model (scaler and DBSCAN) from file
model_data = joblib.load("model2.pkl")
scaler = model_data["scaler"]
dbscan_model = model_data["dbscan"]
eps = dbscan_model.eps
min_samples = dbscan_model.min_samples

@app.post("/predict")
def predict(items: List[ProductData]):
    # Prepare feature array
    X = np.array([[item.avg_price, item.sales_frequency, item.avg_quantity_per_order, item.distinct_customers] for item in items])
    # Scale features
    X_scaled = scaler.transform(X)
    # Perform clustering with same parameters
    clusters = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)
    return {"clusters": clusters.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002) 