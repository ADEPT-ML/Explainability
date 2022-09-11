def calculate_very_basic_feature_attribution(anomaly: int, anomaly_data: dict):
    anomaly_index = anomaly_data["anomalies"][anomaly]["index"]
    return [anomaly_data["error"][i][anomaly_index] / anomaly_data["output_error"][anomaly_index] * 100 for i in range(len(anomaly_data["sensors"]))]