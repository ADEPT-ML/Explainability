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
