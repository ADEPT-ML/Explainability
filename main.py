from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from prototypes import create_prototypes
from feature_attribution import calculate_very_basic_feature_attribution

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "there!"}


@app.post("/prototypes")
def calculate_anomalies(anomaly: int, payload = Body(..., embed=True)):
    a, b, c = create_prototypes(anomaly - 1, payload)
    return {"prototypes": {"prototype a": a,
                           "prototype b": b,
                           "anomaly": c}}

@app.post("/feature-attribution")
def calculate_anomalies(anomaly: int, payload = Body(..., embed=True)):
    attribution = calculate_very_basic_feature_attribution(anomaly - 1, payload)
    attribution = [{"name": payload["sensors"][i], "percent": e} for i, e in enumerate(attribution)]
    # attribution = sorted(attribution, key=lambda x: x["percent"], reverse=True)
    return {"attribution": attribution}