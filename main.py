from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from src import schema, feature_attribution, prototypes

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/prototypes")
def calculate_anomalies(anomaly: int, payload = Body(..., embed=True)):
    a, b, c = prototypes.create_prototypes(anomaly - 1, payload)
    return {"prototypes": {"prototype a": a,
                           "prototype b": b,
                           "anomaly": c}}

@app.post("/feature-attribution")
def calculate_anomalies(anomaly: int, payload = Body(..., embed=True)):
    attribution = feature_attribution.calculate_very_basic_feature_attribution(anomaly - 1, payload)
    attribution = [{"name": payload["sensors"][i], "percent": e} for i, e in enumerate(attribution)]
    # attribution = sorted(attribution, key=lambda x: x["percent"], reverse=True)
    return {"attribution": attribution}


schema.custom_openapi(app)

@app.get(
    "/",
    name="Root path",
    summary="Returns the routes available through the API",
    description="Returns a route list for easier use of API through HATEOAS",
    response_description="List of urls to all available routes",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "payload": [
                            {
                                "path": "/examplePath",
                                "name": "example route"
                            }
                        ]
                    }
                }
            },
        }
    }
)
async def root():
    url_list = [{"path": route.path, "name": route.name}
                for route in app.routes]
    return url_list
