import pandas as pd
import tensorflow as tf
from omnixai.data.timeseries import Timeseries
from omnixai.explainers.timeseries import ShapTimeseries


class ExplainDagmmByUsingSHAP:
    """
    This class implements method to explain anomalies detected by a DAGMM model using SHAP.
    """

    def __init__(self, model, data: pd.DataFrame, anomaly_index):
        """
        :param model: Provide models that require interpretation and use it to construct the prediction function.
        :param data: The data used to initialize the explainer.
        :param anomaly_index: The index used to find anomaly in data and to calculate its shap values.
        """
        self.model = model
        self.idx = anomaly_index
        self.data = data

    def detector(self, ts: Timeseries):
        """
        Generate anomaly scores with the trained model as a predict function.
        :param ts: A `Timeseries' of data, representing an input instance.
        :return: Anomaly scores.
        """
        energy = self.model.predict(tf.keras.utils.timeseries_dataset_from_array(ts.values, None, 96).take(1))
        return energy

    def explain(self):
        """
        Generates the shap values and explanations for the input instances.
        :return: The feature-importance explanations for all the input's features in the form of a percentage .
        """
        train_df = self.data.iloc[self.idx-96:self.idx + 96, :]
        test_df = self.data.iloc[self.idx:self.idx + 96, :]
        test_x = Timeseries.from_pd(test_df)

        explainer = ShapTimeseries(
            training_data=Timeseries.from_pd(train_df),
            predict_function=self.detector,
            mode="anomaly_detection"
        )

        explanations, scores = explainer.explain(test_x, nsamples=100)
        scores = scores.abs()
        res = scores.iloc[-1, :] / scores.iloc[-1, :].sum()

        return res
