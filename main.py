from fastapi import FastAPI, Body, HTTPException, Query
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
    """Root API endpoint that lists all available API endpoints.

    Returns:
        A complete list of all possible API endpoints.
    """
    route_filter = ["openapi", "swagger_ui_html", "swagger_ui_redirect", "redoc_html"]
    url_list = [{"path": route.path, "name": route.name} for route in app.routes if route.name not in route_filter]
    return url_list

@app.post(
    "/prototypes",
    name="Building Sensors",
    summary="Returns a list of sensors of a specified building",
    description="Returns all sensors available for the building specified through the parameter.\
        The response will include a list of the sensors with their type, desc and unit.",
    response_description="List of sensors.",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "prototypes": {
                            "prototype a": [
                                0.01675, 0.01675, 0.01675, 0.01675, 0.07375, 0.07375, 0.07375, 0.07375, 0.0315, 0.0315, 0.0315, 0.0315, 0.049, 0.049, 0.049, 0.049, 0.034, 0.034, 0.034, 0.034, 0.052, 0.052, 0.052, 0.052, 0.063, 0.063, 0.063, 0.063, 0.07175, 0.07175, 0.07175, 0.07175, 0.06775
                            ], 
                            "prototype b": [
                                0.004, 0.004, 0.004, 0.004, 0.00275, 0.00275, 0.00275, 0.00275, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                            ], 
                            "anomaly": [
                                0.0055, 0.0055, 0.0055, 0.0055, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4355, 0.4355, 0.4355, 0.4355, 0.09325, 0.09325, 0.09325, 0.09325, 0.00025, 0.00025, 0.00025, 0.00025, 0.0, 0.0, 0.0, 0.0, 0.0
                            ]
                        }
                    }
                }
            },
        },
        400: {
            "description": "Payload can not be empty.",
            "content": {
                "application/json": {
                    "example": {"detail": "Payload can not be empty"}
                }
            },
        },
        500: {
            "description": "Internal server error.",
            "content": {
                "application/json": {
                    "example": {"detail": "Internal server error"}
                }
            },
        }
    },
    tags=["Buildings and Sensors"]
)
def calculate_anomalies(
    anomaly: int = Query(
        description="Query parameter to select the anomaly.",
        example=0
    ),
    payload = Body(
        default=...,
        description="A dict of the output of anomaly-detection",
        example={
            "payload": {
                "deep-error": [0.03145960019416866, 0.024359986113175414, 0.023060245303469007],
                "dataframe": {
                    "Wasser.1 Diff": {
                        "2020-07-31T20:00:00": 1.4,
                        "2020-07-31T20:15:00": 1.4,
                        "2020-07-31T20:30:00": 1.3
                    },
                    "Electricity.3 Diff": {
                        "2020-07-31T20:00:00": 1.5,
                        "2020-07-31T20:15:00": 1.6,
                        "2020-07-31T20:30:00": 1.7
                    }
                },
                "sensors": ["Wasser.1 Diff", "Elektrizität.1 Diff"],
                "algo": 2,
                "timestamps": ["2020-03-14T11:00:00", "2020-03-14T11:15:00", "2020-03-14T11:30:00"],
                "anomalies": [
                    {"timestamp": "2021-12-21T09:45:00", "type": "Area"}, 
                    {"timestamp": "2021-12-22T09:45:00", "type": "Area"}
                ],
                "error": [0.03145960019416866, 0.024359986113175414, 0.023060245303469007]
            }
        },
        embed=True
    )
):
    try:
        if not payload:
            raise HTTPException(status_code=400, detail="Payload can not be empty")
        a, b, c = prototypes.create_prototypes(anomaly - 1, payload)
        return {"prototypes": {"prototype a": a,
                           "prototype b": b,
                           "anomaly": c}}
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Internal Server Error")
    

@app.post("/feature-attribution")
def calculate_anomalies(anomaly: int, payload = Body(..., embed=True)):
    attribution = feature_attribution.calculate_very_basic_feature_attribution(anomaly - 1, payload)
    attribution = [{"name": payload["sensors"][i], "percent": e} for i, e in enumerate(attribution)]
    # attribution = sorted(attribution, key=lambda x: x["percent"], reverse=True)
    return {"attribution": attribution}


schema.custom_openapi(app)
