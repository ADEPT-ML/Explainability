# Explainability 💡

The service is responsible for all explainability tasks. For example, the computation of prototypes (shapelets,
representations) that better explains anomalous behavior.

## Requirements

+ Python ≥ 3.10
+ All packages from requirements.txt

## Development

### Local

Install dependencies from requirements.txt

Start the service:

```sh
uvicorn main:app --reload
```

### Docker

We provide a docker-compose in the root directory of ADEPT to start all services bundled together.

## Adding functionality

New explainability and feature-attribution methods can easily be added by following the instructions below.
Note, however, that the methods only receive input that can be retrieved from the other services via the API.
Below is an example of the payload you can normally work with.

### Directory structure

```
\-Explainability
    ├── src                                     # Python source files for base functions
    │   ├── feature_attribution.py              # Functions for calculating feature attribution
    │   ├── prototypes.py                       # Functions for calculating explanatory representations
    │   └── [...]
    ├── Dockerfile
    ├── main.py                                 # Main module with all API definitions
    ├── requirements.txt                        # Required python dependencies
    └── [...]
```

### Arguments passed to the endpoint

There are two arguments passed to the `/prototypes` and `/feature-attribution` endpoints:

- `anomaly` __int__ - The index of the anomaly for which the prototypes or attribution are currently
  requested.
  - `anomaly_data` __object__ - The output of the anomaly detection for the given sensor combination
    with all important data. The following example abbreviates lists by `[...]`.
    ```
    {
        "deep-error": [
          [0.01572980009, 0.01217999305, 0.01153012265],
          [0.01572980009, 0.01217999305, 0.01153012265],
          [...]
        ],
        "dataframe": {
          "Wasser.1 Diff": {
              "2020-07-31T20:00:00": 1.4,
              "2020-07-31T20:15:00": 1.4,
              [...]
          },
          "Electricity.1 Diff": {
              "2020-07-31T20:00:00": 1.5,
              "2020-07-31T20:15:00": 1.6,
              [...]
          }
        },
        "sensors": ["Wasser.1 Diff", "Elektrizität.1 Diff"],
        "algo": 2,
        "timestamps": ["2020-03-14T11:00:00", "2020-03-14T11:15:00", [...]],
        "anomalies": [
          {"timestamp": "2021-12-21T09:45:00", "type": "Area"},
          {"timestamp": "2021-12-22T09:45:00", "type": "Area"},
          [...]
        ],
        "error": [0.03145960019416866, 0.024359986113175414, [...]]
    }
    ```

### Adding an explainability method

1. Create a new method in [prototypes.py](src/prototypes.py) with a method-header similar to this one:
   `def create_averaged_prototypes(anomaly: int, anomaly_data: dict, padding: int = 4) -> tuple[list, list, list]:`,
   where...
    1. `anomaly` is the index of the anomaly
    2. `anomaly_data` is the anomaly_data-object
    3. and `padding` is an example of a useful additional argument, in this case used for adding padding to the output
       data
2. Perform calculations with the available data to extract prototypes, patterns or representations and decide on the two
   example windows that best fit the given anomaly
3. Return a tuple containing the two example windows and the anomaly windows, for
   example: `return avg_window, median_window, anomaly_window`

### Adding a feature-attribution method

1. Create a new method in [feature-attribution.py](src/feature_attribution.py) with a method-header similar to this one:
   `def calculate_feature_attribution(anomaly: int, anomaly_data: dict) -> list[float]:`,
   where...
    1. `anomaly` is the index of the anomaly
    2. `anomaly_data` is the anomaly_data-object
2. Calculate the feature-attribution from the available data. Keep in mind: Some feature-attribution methods might
   require additional information that is not available in the `anomaly_data`. For this, you will need to make further
   changes, e.g. to the anomaly detection service and its API.
3. Return the list of percentages of attributions values to the given features. The order of the list should represent
   the order of the features.


Copyright © ADEPT ML, TU Dortmund 2022