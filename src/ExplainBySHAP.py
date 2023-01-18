import json

import pandas as pd
import requests
from .timeseries import Timeseries
from .shap import ShapTimeseries


class ExplainDagmmByUsingSHAP:
    """
    This class implements method to explain anomalies detected by a DAGMM model using SHAP.
    """

    def __init__(self, anomaly_index: int, data: pd.DataFrame, anomaly_data: dict):

        """
        :param data: The data used to initialize the explainer.
        :param anomaly_index: The index used to find anomaly in data and to calculate its shap values.
        :param anomaly_data: The output of the anomaly detection.
        :param length: The length of the selected anomaly.
        """
        self.anomaly_data = anomaly_data
        self.idx = anomaly_index
        self.data = data

    def detector(self, ts: Timeseries):
        """
        Generate anomaly scores with the trained model as a predict function.
        :param ts: A `Timeseries' of data, representing an input instance.
        :return: Anomaly scores.
        """

        algo = self.anomaly_data["algo"]
        building = self.anomaly_data["building"]
        config = self.anomaly_data["config"]
        building_data = pd.DataFrame(ts.values)

        anomaly_url = f"http://anomaly-detection/calculate?algo={algo}&building={building}&config={json.dumps(config)}"
        anomalies_response = requests.post(anomaly_url, json=building_data)
        anomalies = anomalies_response.json()
        return anomalies["error"]

    def explain(self):
        """
        Generates the shap values and explanations for the input instances.
        :return: The feature-importance explanations for all the input's features in the form of a percentage .
        """
        train_df = self.data.iloc[self.idx + 96:self.idx + 388, :]
        test_df = self.data.iloc[self.idx:self.idx + 96, :]
        test_x = Timeseries.from_pd(test_df)

        explainer = ShapTimeseries(
            training_data=Timeseries.from_pd(train_df),
            predict_function=self.detector,
        )

        explanations, scores = explainer.explain(test_x, nsamples=100)
        scores = scores.abs()
        scores.loc['colsum'] = scores.apply(lambda x: x.sum())
        res = scores.iloc[0, :] / scores.iloc[0, :].sum()
        result = np.array(res).tolist()


        return result
