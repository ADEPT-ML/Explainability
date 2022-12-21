"""Contains all functions related to the prototype creation"""
import numpy as np
import pandas as pd

from . import feature_attribution as ft


def create_local_prototypes(anomaly: int, anomaly_data: dict):
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
    sensor = fetch_sensor(anomaly, anomaly_data)
    selected_sensor = sensors[sensor]
    if anomaly_timestamp - two_weeks > df.index[0]:
        a = df.loc[((anomaly_timestamp - two_weeks - time_delta) <= df.index) & (df.index <= (anomaly_timestamp - two_weeks + time_delta)), [selected_sensor]]
        b = df.loc[((anomaly_timestamp - one_week - time_delta) <= df.index) & (df.index <= (anomaly_timestamp - one_week + time_delta)), [selected_sensor]]
    else:
        a = df.loc[((anomaly_timestamp + two_weeks - time_delta) <= df.index) & (df.index <= (anomaly_timestamp + two_weeks + time_delta)), [selected_sensor]]
        b = df.loc[((anomaly_timestamp + one_week - time_delta) <= df.index) & (df.index <= (anomaly_timestamp + one_week + time_delta)), [selected_sensor]]
    c = df.loc[((anomaly_timestamp - time_delta) <= df.index) & (df.index <= (anomaly_timestamp + time_delta)), [selected_sensor]]
    return [e for e in a[selected_sensor]], [e for e in b[selected_sensor]], [e for e in c[selected_sensor]]


def create_averaged_prototypes(anomaly: int, anomaly_data: dict, padding: int = 4):
    """Creates averaged prototypes for the specified anomaly.

    The first two additional timeframes act as an example based explanation for the expected behaviour.
    Fetches the time resolution of the data and supports all resolutions that can represent an hour without a remainder.
    Uses an index based approach to get all related timeframes.
    Calculates one window each with mean and median values.

    Args:
        anomaly: The ID of the anomaly (starting at 0).
        anomaly_data: The output of the anomaly detection.
        padding: The timedelta (in "h") to be used as padding for extending the resulting timeframe on both sides.

    Returns:
        Two averaged prototypes (mean and median) and the anomaly with the same timeframe.
    """
    df = pd.DataFrame(anomaly_data["dataframe"])
    frequency = np.timedelta64(1, "h") // (np.datetime64(df.index[1]) - np.datetime64(df.index[0]))
    padding *= frequency
    week_length = 168 * frequency
    anomaly_index = anomaly_data["anomalies"][anomaly]["index"]
    anomaly_length = anomaly_data["anomalies"][anomaly]["length"]
    anomaly_low_bound = anomaly_index - padding
    low_bound = anomaly_low_bound % week_length
    w_length = 2 * padding + anomaly_length
    sensor = fetch_sensor(anomaly, anomaly_data)

    df = df.loc[:, anomaly_data["sensors"][sensor]]
    indices = range(low_bound, len(anomaly_data["timestamps"]) - w_length, week_length)
    windows = np.swapaxes(np.array([df.iloc[i:i + w_length] for i in indices]), 0, 1)

    avg_window = [np.average(e) for e in windows]
    median_window = [np.median(e) for e in windows]
    anomaly_window = [None] * abs(anomaly_low_bound) if anomaly_low_bound < 0 else []
    anomaly_window.extend(df.iloc[max(anomaly_low_bound, 0):min(anomaly_low_bound + w_length, len(df.index))].tolist())
    if anomaly_low_bound + w_length > len(df.index):
        anomaly_window.extend([None] * anomaly_low_bound + w_length - len(df.index))
    return avg_window, median_window, anomaly_window


def create_averaged_prototypes_mask(anomaly: int, anomaly_data: dict, padding: int = 4):
    """Creates averaged prototypes for the specified anomaly.

    A mask with similar timeframes (based on the day and time) is applied to the dataframe
    and the remaining values are averaged to create an explanatory example for expected behavior

    Args:
        anomaly: The ID of the anomaly.
        anomaly_data: The output of the anomaly detection.
        padding (default=4): The timedelta (in 'h') to be used as padding for extending the returned timeframe. 

    Returns:
        Two averaged prototypes (mean and median) and the anomaly with the same timeframe.
    """
    sensors = anomaly_data["sensors"]
    anomaly_timestamp = np.datetime64(anomaly_data["timestamps"][anomaly_data["anomalies"][anomaly]["index"]])
    anomaly_span = anomaly_data["anomalies"][anomaly]["length"]
    
    df = pd.DataFrame(anomaly_data["dataframe"])
    df.index = pd.to_datetime(df.index.values)

    # calculate the timedelta between two tuples and multiply by anomaly-length to get timeframe of anomaly
    time_diff = df.iloc[:2].index.to_series().diff().astype('timedelta64[m]')[-1]
    anomaly_length = ((anomaly_span-1)*time_diff).astype('timedelta64[m]')
    time_padding = np.timedelta64(padding, 'h')

    sensor = fetch_sensor(anomaly, anomaly_data)
    selected_sensor = sensors[sensor]
    
    time_start = (anomaly_timestamp - time_padding).astype(object)
    weekday_start = time_start.weekday()
    time_start_minutes = (time_start.hour+(weekday_start*24))*60+time_start.minute
    time_end = (anomaly_timestamp + anomaly_length + time_padding).astype(object)
    weekday_end = time_end.weekday()
    time_end_minutes = (time_end.hour+(weekday_end*24))*60+time_end.minute
    
    # mask data with the same weekday(s) in the same period of time +- timedelta
    a = df.loc[
        ((weekday_end >= weekday_start) &
            ((df.index.weekday >= weekday_start) & 
             (df.index.weekday <= weekday_end) &
             (((df.index.hour+(df.index.weekday*24))*60+df.index.minute >= time_start_minutes) &
              ((df.index.hour+(df.index.weekday*24))*60+df.index.minute <= time_end_minutes))))
        |
        ((time_start.isocalendar().week < time_end.isocalendar().week) &  # TODO: year can also be different
            (((df.index.weekday >= weekday_start) &
             ((df.index.hour+(df.index.weekday*24))*60+df.index.minute >= time_start_minutes))
            |
            (((df.index.weekday <= weekday_end) &
             ((df.index.hour+(df.index.weekday*24))*60+df.index.minute <= time_end_minutes))))),
        [selected_sensor]
    ]
    
    np_a = a.to_numpy()
    len_frame = 2 * padding * (np.timedelta64(1, 'h') // time_diff.astype('timedelta64[m]')) + anomaly_span
    np_a = np.swapaxes(np.reshape(np_a, (len(np_a) // len_frame, len_frame)), 0, 1)
    
    a = [np.average(e) for e in np_a]
    b = [np.median(e) for e in np_a]
    c = df.loc[((anomaly_timestamp - time_padding) <= df.index) & (df.index <= (anomaly_timestamp + anomaly_length + time_padding)), [selected_sensor]]

    return a, b, [e for e in c[selected_sensor]]


def fetch_sensor(anomaly, anomaly_data):
    """Determines the sensor for the prototype creation.

    If no values for the feature attribution are present the first sensor will be returned.

    Args:
        anomaly: The ID of the anomaly (starting at 0).
        anomaly_data: The output of the anomaly detection.

    Returns:
        The sensor responsible for the anomaly or the first if no feature attribution data is present.
    """
    if anomaly_data["deep-error"]:
        feature_attribution = ft.calculate_averaged_feature_attribution(anomaly, anomaly_data)
        return feature_attribution.index(max(feature_attribution))
    else:
        return 0
