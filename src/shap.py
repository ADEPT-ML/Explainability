import shap
import numpy as np
import pandas as pd
from omnixai.data.timeseries import Timeseries


class ShapTimeseries():
    """
    The SHAP explainer for time series forecasting and anomaly detection.
    If using this explainer, please cite the original work: https://github.com/slundberg/shap.
    """

    def __init__(
            self,
            training_data: Timeseries,
            predict_function,
    ):
        """
        :param training_data: The data used to initialize the explainer.
        :param predict_function: The prediction function corresponding to the model to explain.
            The input of ``predict_function`` is an `Timeseries` instance. The output of ``predict_function``
            is the anomaly score (higher scores imply more anomalous) for anomaly detection or the predicted
            value for forecasting.
        """
        super().__init__()
        assert isinstance(training_data, Timeseries), \
            "`training_data` should be an instance of Timeseries."

        self.data = training_data
        self.predict_function = predict_function
        self.variables = list(self.data.columns)

        # The lengths of test instances must be the same
        self.explainer = None
        self.test_ts_length = None
        self.index2timestamps = None
        self.all_idx2ts = None

    def _build_data(self, ts_len, num_samples):
        interval = range(self.data.ts_len - ts_len)
        ps = random.sample(interval, min(num_samples, len(interval)))

        # Data for initializing the model
        ps = [interval[0], interval[-1]]
        samples, index2timestamps = [], {}
        for i, p in enumerate(ps):
            index2timestamps[i] = self.data.index[p:p + ts_len]
            x = self.data.values[p:p + ts_len]
            y = np.zeros((x.shape[0] * x.shape[1] + 1,))
            y[:-1], y[-1] = x.flatten(), i
            # y = np.zeros((x.shape[0] * x.shape[1],))
            # y = x.flatten()
            samples.append(y.flatten())
        return np.array(samples), index2timestamps

    def _build_predictor(self, ts_len):
        def _predict(xs: np.ndarray):
            xs = xs.reshape((xs.shape[0], -1))
            ts = [
                Timeseries(
                    data=x[:-1].reshape((ts_len, -1)),
                    variable_names=self.variables,
                    timestamps=self.all_idx2ts[x[-1]]
                ) for x in xs]
            try:
                return np.array(self.predict_function(ts)).flatten()
            except:
                return np.array([self.predict_function(t) for t in ts]).flatten()

        return _predict

    def _build_explainer(self, ts_len, num_samples=100):
        if self.explainer is not None:
            return
        assert self.data.ts_len > ts_len, \
            "`ts_length` should be less than the length of the training time series"

        data, self.index2timestamps = self._build_data(ts_len, num_samples)
        self.all_idx2ts = self.index2timestamps.copy()
        self.explainer = shap.KernelExplainer(
            model=self._build_predictor(ts_len),
            data=data,
            link="identity"
        )
        self.test_ts_length = ts_len

    def explain(self, X: Timeseries, **kwargs):
        """
        Generates the feature-importance explanations for the input instances.

        :param X: An instance of `Timeseries` representing one input instance or
            a batch of input instances.
        :param kwargs: Additional parameters for `shap.KernelExplainer.shap_values`, e.g.,
            "nsamples" for the number of times to re-evaluate the model when explaining each prediction.
        :return: The feature-importance explanations for all the input instances.
        """
        # Initialize the SHAP explainer if it is not created.
        self._build_explainer(X.ts_len)

        index = max(self.index2timestamps.keys()) + 1
        self.all_idx2ts = self.index2timestamps.copy()
        self.all_idx2ts[index] = X.index
        # instances = np.zeros((1, X.shape[0] * X.shape[1]))
        # instances[:, :] = X.values.reshape((1, -1))
        instances = np.zeros((1, X.shape[0] * X.shape[1] + 1))
        instances[:, :-1] = X.values.reshape((1, -1))
        instances[:, -1] = index

        shap_values = self.explainer.shap_values(instances, l1_reg=False, **kwargs)
        shap_values = shap_values.flatten()
        # metric_shap_values = shap_values.reshape(X.shape)
        metric_shap_values = shap_values[:-1].reshape(X.shape)
        timestamp_shap_value = shap_values[-1]

        scores = pd.DataFrame(
            metric_shap_values,
            columns=X.columns,
            index=X.index
        )
        # scores["@timestamp"] = timestamp_shap_value
        return scores
