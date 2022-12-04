"""The main module with all API definitions of the Explainability service"""
from fastapi import FastAPI, Body, HTTPException, Query

from src import schema, feature_attribution, prototypes


app = FastAPI()


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
    name="Get prototypes for a selected anomaly",
    summary="Get the prototypes for a selected anomaly",
    description="Returns a dict with two prototypes and the original anomaly.",
    response_description="Dict of prototypes and anomalies.",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "prototypes": {
                            "prototype a": [
                                0.01675, 0.01675, 0.01675, 0.01675, 0.07375, 0.07375, 0.07375, 0.07375, 0.0315, 0.0315,
                                0.0315, 0.0315, 0.049, 0.049, 0.049, 0.049, 0.034, 0.034, 0.034, 0.034, 0.052, 0.052,
                                0.052, 0.052, 0.063, 0.063, 0.063, 0.063, 0.07175, 0.07175, 0.07175, 0.07175, 0.06775
                            ],
                            "prototype b": [
                                0.004, 0.004, 0.004, 0.004, 0.00275, 0.00275, 0.00275, 0.00275, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0
                            ],
                            "anomaly": [
                                0.0055, 0.0055, 0.0055, 0.0055, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.4355, 0.4355, 0.4355, 0.4355, 0.09325, 0.09325, 0.09325, 0.09325, 0.00025,
                                0.00025, 0.00025, 0.00025, 0.0, 0.0, 0.0, 0.0, 0.0
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
    tags=["Prototypes"]
)
def calculate_prototypes(
        anomaly: int = Query(
            description="Query parameter to select the anomaly.",
            example=0
        ),
        payload=Body(
            default=...,
            description="A dict of the output of anomaly-detection",
            example={
                "payload": {
                    "deep-error": [
                        [0.01572980009, 0.01217999305, 0.01153012265],
                        [0.01572980009, 0.01217999305, 0.01153012265]
                    ],
                    "dataframe": {
                        "Wasser.1 Diff": {
                            "2020-07-31T20:00:00": 1.4,
                            "2020-07-31T20:15:00": 1.4,
                            "2020-07-31T20:30:00": 1.3
                        },
                        "Electricity.1 Diff": {
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
    """Creates prototypes for the specified anomaly.

    Args:
        anomaly: The ID of the anomaly for which the prototypes are created.
        payload: The output of the anomaly detection.

    Returns:
        Two created prototypes and the anomaly with the same timeframe.
    """
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


@app.post(
    "/feature-attribution",
    name="Get attribution of features for a selected anomaly",
    summary="Get the attribution of features for a selected anomaly",
    description="Returns a a list with the names and percentages of the feature attribution.",
    response_description="A list with the names and percentages of feature attribution.",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "attribution": [
                            {'name': 'Wasser.1 Diff', 'percent': 82.65603968422548},
                            {'name': 'Elektrizität.1 Diff', 'percent': 17.343960315774527}
                        ]
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
    tags=["Attributions"]
)
def calculate_attribution(
        anomaly: int = Query(
            description="Query parameter to select the anomaly.",
            example=0
        ),
        payload=Body(
            default=...,
            description="A dict of the output of anomaly-detection",
            example={
                "payload": {
                    "deep-error": [
                        [0.01572980009, 0.01217999305, 0.01153012265],
                        [0.01572980009, 0.01217999305, 0.01153012265]
                    ],
                    "dataframe": {
                        "Wasser.1 Diff": {
                            "2020-07-31T20:00:00": 1.4,
                            "2020-07-31T20:15:00": 1.4,
                            "2020-07-31T20:30:00": 1.3
                        },
                        "Electricity.1 Diff": {
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
    """Calculates the feature attribution for the specified anomaly.

    Args:
        anomaly: The ID of the anomaly for which the prototypes are created.
        payload: The output of the anomaly detection.

    Returns:
        The calculated feature attribution for the specified anomaly.
    """
    try:
        attribution = feature_attribution.calculate_very_basic_feature_attribution(anomaly - 1, payload)
        attribution = [{"name": payload["sensors"][i], "percent": e} for i, e in enumerate(attribution)]
        # attribution = sorted(attribution, key=lambda x: x["percent"], reverse=True)
        return {"attribution": attribution}
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Internal Server Error")


schema.custom_openapi(app)
