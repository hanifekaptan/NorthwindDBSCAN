from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
from sklearn.cluster import DBSCAN

class CustomerData(BaseModel):
    total_orders: int
    total_spent: float
    avg_order_value: float

    class Config:
        json_schema_extra = {
            "example":
                {
                    "total_orders": 10,
                    "total_spent": 1250.75,
                    "avg_order_value": 125.08
                }
        }

app = FastAPI(title="Customer Segmentation API")

# Load trained model (scaler and DBSCAN) from file
model_data = joblib.load("model1.pkl")
scaler = model_data["scaler"]
dbscan_model = model_data["dbscan"]
eps = dbscan_model.eps
min_samples = dbscan_model.min_samples

@app.post("/predict")
def predict(items: List[CustomerData]):
    # Prepare feature array
    X = np.array([[item.total_orders, item.total_spent, item.avg_order_value] for item in items])
    # Scale features
    X_scaled = scaler.transform(X)
    # Perform clustering with same parameters
    clusters = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)
    return {"clusters": clusters.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001) 