from fastapi import FastAPI
from app.routers import predict, health

app = FastAPI(title="Credit Card Fraud Detection API")

app.include_router(health.router)
app.include_router(predict.router)

@app.get("/")
def root():
    return {"message": "Welcome to the Fraud Detection API"}
