"""Contains all functions related to the feature attribution"""


def calculate_very_basic_feature_attribution(anomaly: int, anomaly_data: dict):
    """Calculates a feature attribution based on the specified anomaly and the output of the anomaly detection.

    Args:
        anomaly: The ID of the anomaly.
        anomaly_data: The output of the anomaly detection.

    Returns:
        A percentage for each feature that determines its influence on the detected anomaly.
    """
    anomaly_index = anomaly_data["anomalies"][anomaly]["index"]
    return [anomaly_data["deep-error"][i][anomaly_index] / anomaly_data["error"][anomaly_index] * 100 for i in range(len(anomaly_data["sensors"]))]

def calculate_basic_feature_attribution(anomaly: int, anomaly_data: dict):
    """Calculates a feature attribution based on the specified anomaly and the output of the anomaly detection.

    Uses the middle of the anomaly area for the feature attribution

    Args:
        anomaly: The ID of the anomaly.
        anomaly_data: The output of the anomaly detection.

    Returns:
        A percentage for each feature that determines its influence on the detected anomaly.
    """
    anomaly_index = anomaly_data["anomalies"][anomaly]["index"] + anomaly_data["anomalies"][anomaly]["length"] // 2
    return [anomaly_data["deep-error"][i][anomaly_index] / anomaly_data["error"][anomaly_index] * 100 for i in range(len(anomaly_data["sensors"]))]


def calculate_averaged_feature_attribution(anomaly: int, anomaly_data: dict):
    """Calculates results feature attribution based on the specified anomaly and the output of the anomaly detection.

    Uses the averaged results of the anomaly area for the feature attribution

    Args:
        anomaly: The ID of the anomaly.
        anomaly_data: The output of the anomaly detection.

    Returns:
        A percentage for each feature that determines its influence on the detected anomaly.
    """
    anomaly_index = anomaly_data["anomalies"][anomaly]["index"]
    anomaly_length = anomaly_data["anomalies"][anomaly]["length"]
    results = []
    for i in range(len(anomaly_data["sensors"])):
        sensor_percentages = []
        for j in range(anomaly_index, anomaly_index + anomaly_length):
            sensor_percentages.append(anomaly_data["deep-error"][i][j])
        results.append(sum(sensor_percentages) / len(sensor_percentages))
    return [(e / sum(results)) * 100 for e in results]


def calculate_median_feature_attribution(anomaly: int, anomaly_data: dict):
    """Calculates a feature attribution based on the specified anomaly and the output of the anomaly detection.

    Uses the median values of the anomaly area for the feature attribution

    Args:
        anomaly: The ID of the anomaly.
        anomaly_data: The output of the anomaly detection.

    Returns:
        A percentage for each feature that determines its influence on the detected anomaly.
    """
    anomaly_index = anomaly_data["anomalies"][anomaly]["index"]
    anomaly_length = anomaly_data["anomalies"][anomaly]["length"]
    results = []
    for i in range(len(anomaly_data["sensors"])):
        sensor_percentages = []
        for j in range(anomaly_index, anomaly_index + anomaly_length):
            sensor_percentages.append(anomaly_data["deep-error"][i][j])
        results.append(sorted(sensor_percentages)[len(sensor_percentages) // 2])
    return [(e / sum(results)) * 100 for e in results]
