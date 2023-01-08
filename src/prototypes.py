# """Contains all functions related to the prototype creation"""
import numpy as np
import pandas as pd
import stumpy as sp
from src import feature_attribution as ft

 
def create_shapelets(anomaly: int, anomaly_data: dict):
    """Searches the shapelets of the normal state for the specific anomaly.
    
    Finds two shapelets or motifs that describe the normal state. Shapelets are found with help 
    of matirx profiles calculated with "stump" function. For the motifs the minimum values are 
    considered in the matrix profiles. 
    
    Args:
        anomaly (int): The ID of the anomaly.
        anomaly_data (dict): The output of the anomaly detection.

    Returns:
        Two found shapelets and the anomaly with the same length.
    """
    padding = 4
    matrixProfile = []
    indexes_max_distance = []
    shapelets = []
    building_df = pd.DataFrame(anomaly_data["dataframe"])
    building_df.index = pd.to_datetime(building_df.index.values)
    frequency = np.timedelta64(1, "h") // (np.datetime64(building_df.index[1]) - np.datetime64(building_df.index[0]))
    padding *= frequency
    anomaly_length = anomaly_data["anomalies"][anomaly]["length"]
    w_length = 2 * padding + anomaly_length
    w_length_int = int(w_length)
    keys = building_df.keys()
    
    for key in keys:
        sensor_df = pd.DataFrame(building_df[key]).reset_index()
        matrixProfile = find_matrix_profile(building_df[key], w_length_int)
        indexes_max_distance = find_min_distance(matrixProfile)
        for index_max_distance in indexes_max_distance:
            shapelet = []
            for a in range(w_length_int):
                shapelet.append(sensor_df.loc[index_max_distance + a, key])
            shapelets.append(shapelet)    
    shapelets_df = shapelet_to_df(shapelets)

    result = min_distance_to_anomaly(anomaly, anomaly_data, building_df, shapelets_df, w_length_int, padding)
    best_shapelet_names = result[0]
    anomaly_as_list = result[1]
    first_shapelet = [e for e in shapelets_df[best_shapelet_names[0]]]
    second_shapelet = [e for e in shapelets_df[best_shapelet_names[1]]]
    
    return first_shapelet, second_shapelet, anomaly_as_list
        
def find_matrix_profile(building_sensor: pd.Series, window: int, secondSeries=None):
    """Finds matrix profile.

    Args:
        building_sensor (pd.Series): Data of the secific sensor of the building.
        window (int): Size of the window.

    Returns:
        Matrix profile as an array.
    """
    if(secondSeries is not None):
        matrixProfile = sp.stump(building_sensor, window, secondSeries)[:, 0].astype(float)
        matrixProfile[matrixProfile == np.inf] = np.nan
    else:
        matrixProfile = sp.stump(building_sensor, window)[:, 0].astype(float)
        matrixProfile[matrixProfile == np.inf] = np.nan
    
    return matrixProfile

def find_min_distance(matrixProfile: np.ndarray):
    """Searches for the top 12 minimum values in matrix profile.

    Args:
        matrixProfile (np.ndarray): Array with matrix profile.

    Returns:
        Array of indexes of 12 minimum values.
    """
    indexes_min_dist = []

    for i in range(12):
        currmin = float('inf')
        currminid = 0
        for index, value in enumerate(matrixProfile):
            if value < currmin:
                currmin = value
                currminid = index
        matrixProfile[currminid] = float('inf')
        indexes_min_dist.append(currminid)
        
    return indexes_min_dist

def shapelet_to_df(shapelets: np.ndarray):
    """Converts array of shapelst into DataFrame.

    Args:
        shapelets (np.ndarray): Array of shapelets.

    Returns:
        DataFrame of shapelets.
    """
    shapelets_df = pd.DataFrame(shapelets[0], columns=['shapelet'+str(0)]) 

    for shapelet in range(len(shapelets)-1):
        column = "shapelet" + str(shapelet+1)
        shapelets_df[column] = pd.DataFrame(shapelets[shapelet+1])
    return shapelets_df

def euclidean_dist(series1: pd.Series, series2: pd.Series):
    """Finds Euclidean Distance between two series.

    Args:
        series1 (pd.Series):
        series2 (pd.Series):

    Returns:
        Euclidean Distance.
    """
    return np.linalg.norm(series1.values - series2.values)

def min_distance_to_anomaly(anomaly: int, anomaly_data: dict, building_df:pd.DataFrame, shapelets_df:pd.DataFrame, window:int, padding:np.longlong):
    """Finds the most suitable two shapelets.
    
    Searches for the most suitable shapelets by calculating z-normalized 
    euclidean distance between shapelet and anomaly. Two shaplets with
    the biggest z-normalised distance are chosen.

    Args:
        anomaly (int): The ID of the anomaly.
        anomaly_data (dict): The output of the anomaly detection.
        building_df (pd.DataFrame): The DataFrame of data of chosen building.
        shapelets_df (pd.DataFrame): The DataFrame with all shapelets.

    Returns:
        Names of the two most suitable shapelets and anomaly as a list.
    """
    maxim = 0
    best_shapelets = []
    best_shapelet_names = []
    
    sensor = fetch_sensor(anomaly, anomaly_data)
    building_df = building_df.loc[:, anomaly_data["sensors"][sensor]]
    anomaly_index = anomaly_data["anomalies"][anomaly]["index"]
    anomaly_low_bound = anomaly_index - padding
    anomaly_window = [None] * abs(anomaly_low_bound) if anomaly_low_bound < 0 else []
    anomaly_window.extend(building_df.iloc[max(anomaly_low_bound, 0):min(anomaly_low_bound + window, len(building_df.index))])
    anomaly_window_series = pd.Series(anomaly_window)
    
    #----z-normalized euclidean distance option
    for shapelet in shapelets_df:
        zdist = find_matrix_profile(shapelets_df[shapelet], 3, anomaly_window_series)
        sums = sum(zdist)
        best_shapelets.append((sums, shapelet))
        
    #----euclidean distance option 
    #for shapelet in shapelets_df:
        #edist = euclidean_dist(shapelets_df[shapelet], anomaly_window_series)
        #best_shapelets.append((edist, shapelet))
        
    for i in range(2):
        for best_shapelet in best_shapelets:
            if best_shapelet[0] >= maxim:
                maxim = best_shapelet[0]
                best = best_shapelet[1]       
        best_shapelet_names.append(best)
        best_shapelets.remove((maxim, best))
        maxim = 0
    
    return best_shapelet_names, anomaly_window

def create_local_prototypes(anomaly: int, anomaly_data: dict) -> tuple[list, list, list]:
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


def create_averaged_prototypes(anomaly: int, anomaly_data: dict, padding: int = 4) -> tuple[list, list, list]:
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


def create_averaged_prototypes_mask(anomaly: int, anomaly_data: dict, padding: int = 4) -> tuple[list, list, list]:
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


def fetch_sensor(anomaly, anomaly_data) -> int:
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
