from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
from sklearn.cluster import DBSCAN

class SupplierData(BaseModel):
    num_products_supplied: int
    total_sales_quantity: float
    avg_selling_price: float
    total_distinct_customers: int

    class Config:
        json_schema_extra = {
            "example":
                {
                    "num_products_supplied": 5,
                    "total_sales_quantity": 2500.0,
                    "avg_selling_price": 45.75,
                    "total_distinct_customers": 120
                }
        }

app = FastAPI(title="Supplier Segmentation API")

# Load trained model (scaler and DBSCAN) from file
model_data = joblib.load("model3.pkl")
scaler = model_data["scaler"]
dbscan_model = model_data["dbscan"]
eps = dbscan_model.eps
min_samples = dbscan_model.min_samples

@app.post("/predict")
def predict(items: List[SupplierData]):
    # Prepare feature array
    X = np.array([[
        item.num_products_supplied,
        item.total_sales_quantity,
        item.avg_selling_price,
        item.total_distinct_customers
    ] for item in items])
    # Scale features
    X_scaled = scaler.transform(X)
    # Perform clustering with same parameters
    clusters = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)
    return {"clusters": clusters.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003) 