# """Contains all functions related to the prototype creation"""
import numpy as np
import pandas as pd
import stumpy as sp
from src import feature_attribution as ft

 
def create_shapelets(anomaly: int, anomaly_data: dict):
    """Searches the shapelets of the normal state for the specific anomaly.
    Args:
        anomaly: The ID of the anomaly.
        anomaly_data: The output of the anomaly detection.

    Returns:
        Two found shapelets and the anomaly with the same length.
    """
    window_length = 32
    matrixProfile = []
    indexes_max_distance = []
    shapelets = []
    
    building_df = pd.DataFrame(anomaly_data["dataframe"])
    building_df.index = pd.to_datetime(building_df.index.values)
    keys = building_df.keys()
    
    for key in keys:
        sensor_df = pd.DataFrame(building_df[key]).reset_index()
        matrixProfile = find_matrix_profile(building_df[key], window_length)
        indexes_max_distance = find_max_distance(matrixProfile)
        for index_max_distance in indexes_max_distance:
            shapelet = []
            for a in range(window_length+1):
                shapelet.append(sensor_df.loc[index_max_distance + a, key])
            shapelets.append(shapelet)
          
    shapelets_df = shapelet_to_df(shapelets)

    result = min_distance_to_anomaly(anomaly, anomaly_data, shapelets_df)
    best_shapelet_names = result[0]
    anomaly_as_list = result[1]
    first_shapelet = [e for e in shapelets_df[best_shapelet_names[0]]]
    second_shapelet = [e for e in shapelets_df[best_shapelet_names[1]]]
    
    return first_shapelet, second_shapelet, anomaly_as_list

        
def find_matrix_profile(building_sensor: pd.Series, window: int):
    """Finds matrix profile.

    Args:
        building_sensor (pd.Series): Data of the secific sensor of the building.
        window (int): Size of the window.

    Returns:
        Matrix profile as an array.
    """
    matrixProfile = sp.stump(building_sensor, window)[:, 0].astype(float)
    matrixProfile[matrixProfile == np.inf] = np.nan
    
    return matrixProfile







def find_min_distance(matrixProfile: np.ndarray):
    """Searches for the top 5 max values in matrix profile.

    Args:
        matrixProfile (np.ndarray): Array with matrix profile.

    Returns:
        Array of indexes of the top 5 values with the most difference.
    """
    indexes_min_dist = []

    for i in range(5):
        currmin = float('inf')
        currminid = 0
        for index, value in enumerate(matrixProfile):
            if value < currmin:
                currmin = value
                currminid = index
        matrixProfile[currminid] = float('inf')
        indexes_min_dist.append(currminid)
        
    return indexes_min_dist








def find_max_distance(matrixProfile: np.ndarray):
    """Searches for the top 5 max values in matrix profile.

    Args:
        matrixProfile (np.ndarray): Array with matrix profile.

    Returns:
        Array of indexes of the top 5 values with the most difference.
    """
    indexes_max_dist = []

    for i in range(5):
        currentMaximum = 0
        current_max_ind = 0
        for index, value in enumerate(matrixProfile):
            if value > currentMaximum:
                currentMaximum = value
                current_max_ind = index
        matrixProfile[current_max_ind] = 0
        indexes_max_dist.append(current_max_ind)
        
    return indexes_max_dist

def shapelet_to_df(shapelets: np.ndarray):
    """Converts array of shapelst into DataFrame.

    Args:
        shapelets: Array of shapelets.

    Returns:
        DataFrame of shapelets.
    """
    shapelets_df = pd.DataFrame(shapelets[0], columns=['shapelet'+str(0)]) 

    for shapelet in range(len(shapelets)-1):
        column = "shapelet" + str(shapelet+1)
        shapelets_df[column] = pd.DataFrame(shapelets[shapelet+1])
    return shapelets_df

def euclidean_dist(series1: pd.Series, series2: pd.Series):
    """Finds Euclidean Distance.

    Args:
        series1 (pd.Series):
        series2 (pd.Series):

    Returns:
        Euclidean Distance.
    """
    return np.linalg.norm(series1.values - series2.values)

def min_distance_to_anomaly(anomaly: int, anomaly_data: dict, shapelets_df: pd.DataFrame):
    """Finds minimal distance of a shapelet to anomaly without considering an anomaly itself. 

    Args:
        anomaly: The ID of the anomaly.
        anomaly_data: The output of the anomaly detection.
        shapelets_df: The DataFrame with all shapelets.

    Returns:
        Names of the two most suitable shapelets and anomaly as a list.
    """
    min = float('inf')
    best_shapelets = []
    best_shapelet_names = []
    sensors = anomaly_data["sensors"]
    anomaly_timestamp = np.datetime64(anomaly_data["timestamps"][anomaly_data["anomalies"][anomaly]["index"]])
    df = pd.DataFrame(anomaly_data["dataframe"])
    df.index = pd.to_datetime(df.index.values)
    time_delta = np.timedelta64(4, 'h')
    
    if anomaly_data["algo"] == 2:
        feature_attribution = ft.calculate_very_basic_feature_attribution(anomaly, anomaly_data)
        sensor = feature_attribution.index(max(feature_attribution))
    else:
        sensor = 0
    selected_sensor = sensors[sensor]

    anomaly_df = df.loc[((anomaly_timestamp - time_delta) <= df.index) & (df.index <= (anomaly_timestamp + time_delta)), [selected_sensor]]
    anomaly_normal = anomaly_df
    anomaly_normal.loc[anomaly_timestamp, selected_sensor] = anomaly_df[selected_sensor].mean(axis=0)
    
    for shapelet in shapelets_df:
        edist = euclidean_dist(shapelets_df[shapelet], anomaly_normal[selected_sensor])
        best_shapelets.append((edist, shapelet))
    
    for i in range(2):
        print(i)
        for best_shapelet in best_shapelets:
            if best_shapelet[0] <= min:
                print(min)
                min = best_shapelet[0]
                best = best_shapelet[1]
        print("vor", best_shapelets)        
        best_shapelet_names.append(best)
        best_shapelets.remove((min, best))
        min = float('inf')
        print("nach", best_shapelets)
    
    anomaly_as_list = [e for e in anomaly_df[selected_sensor]]
    
    return best_shapelet_names, anomaly_as_list


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
    x = [e for e in a[selected_sensor]]
    y = [e for e in b[selected_sensor]]
    z = [e for e in c[selected_sensor]]
    return x, y, z