from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from app.services.storage import load_forecast
from app.services.predictor import update_forecast_file
from app.services.regions import load_regions

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",

        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Backend is working"}


@app.get("/forecast")
def get_forecast(region: str = Query(default="all")):
    data = load_forecast()

    if region.lower() == "all":
        return data

    region_data = data.get("regions_forecast", {}).get(region)

    if region_data is None:
        return {"error": f"Region '{region}' not found"}

    return {
        "last_model_train_time": data.get("last_model_train_time"),
        "last_prediction_time": data.get("last_prediction_time"),
        "model_name": data.get("model_name"),
        "forecast_horizon_hours": data.get("forecast_horizon_hours"),
        "regions_forecast": {
            region: region_data
        }
    }

@app.get("/regions")
def get_regions():
    return load_regions()

@app.post("/forecast/update")
def update_forecast():
    new_forecast = update_forecast_file()
    return {
        "message": "Forecast updated successfully",
        "data": new_forecast
    }
