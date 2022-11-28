"""Contains all functions related to the prototype creation"""
import numpy as np
import pandas as pd
from src import feature_attribution as ft

def create_prototypes(anomaly: int, anomaly_data: dict):
    """Creates prototypes for the specified anomaly.

    Generates two similar timeframes based on the surrounding weeks.
    Adds the timeframe of the anomaly itself.

    Args:
        anomaly: The ID of the anomaly.
        anomaly_data: The output of the anomaly detection.

    Returns:
        Two created prototypes and the anomaly with the same timeframe.
    """
    sensors = anomaly_data["sensors"]
    anomaly_timestamp = np.datetime64(anomaly_data["timestamps"][anomaly_data["anomalies"][anomaly]["index"]])
    df = pd.DataFrame(anomaly_data["dataframe"])
    df.index = pd.to_datetime(df.index.values)
    time_delta = np.timedelta64(4, 'h')
    one_week = np.timedelta64(7, 'D')
    two_weeks = np.timedelta64(14, 'D')
    if anomaly_data["algo"] == 2:
        feature_attribution = ft.calculate_very_basic_feature_attribution(anomaly, anomaly_data)
        sensor = feature_attribution.index(max(feature_attribution))
    else:
        sensor = 0
    selected_sensor = sensors[sensor]
    if anomaly_timestamp - two_weeks > df.index[0]:
        a = df.loc[((anomaly_timestamp - two_weeks - time_delta) <= df.index) & (df.index <= (anomaly_timestamp - two_weeks + time_delta)), [selected_sensor]]
        b = df.loc[((anomaly_timestamp - one_week - time_delta) <= df.index) & (df.index <= (anomaly_timestamp - one_week + time_delta)), [selected_sensor]]
    else:
        a = df.loc[((anomaly_timestamp + two_weeks - time_delta) <= df.index) & (df.index <= (anomaly_timestamp + two_weeks + time_delta)), [selected_sensor]]
        b = df.loc[((anomaly_timestamp + one_week - time_delta) <= df.index) & (df.index <= (anomaly_timestamp + one_week + time_delta)), [selected_sensor]]
    c = df.loc[((anomaly_timestamp - time_delta) <= df.index) & (df.index <= (anomaly_timestamp + time_delta)), [selected_sensor]]
    return [e for e in a[selected_sensor]], [e for e in b[selected_sensor]], [e for e in c[selected_sensor]]