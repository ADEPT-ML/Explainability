def calculate_very_basic_feature_attribution(anomaly: int, anomaly_data: dict):
    anomaly_index = anomaly_data["anomalies"][anomaly]["index"]
    anomaly_amount = len(anomaly_data["sensors"])
    output_data = []
    for i in range(anomaly_amount):
        calc_data = anomaly_data["deep-error"][i][anomaly_index] / anomaly_data["error"][anomaly_index] * 100
        output_data.append(calc_data)
    return output_data