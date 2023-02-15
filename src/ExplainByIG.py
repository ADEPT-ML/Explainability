import json
import math
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients


class LSTMAI():
    "Used to calculate the Integrated Gradient"

    def Interpreter(self, errors, mean, cov):
        """Calculate its feature contribution by the anomalies found by the LSTM Autoencoder model.
        :param errors: The anomaly's construction errors that LSTM calculates.
        :param mean: In training LSTM Autoencoder, the mean value of the overall training data obtained.
        :param cov: In training LSTM Autoencoder, the covariance value of the overall training data obtained.
        :return: Feature Attributions.
        """

        mean = torch.tensor(mean)
        cov = torch.tensor(cov)

        # MAHALANOBIS DISTANCE
        def wrapper(x):
            x = x.view(-1, x.shape[2])
            deviations = x - mean
            inverse = torch.inverse(cov).float()

            m = torch.matmul(torch.matmul(deviations, inverse).float(), deviations.t())
            n = torch.sqrt(m)

            res = torch.sum(n, dim=1)
            return res
        attributions = []

        explainer = IntegratedGradients(forward_func=wrapper)

        for e in errors:
            e = e.view(1, -1, e.shape[0])
            attribution = explainer.attribute(e, n_steps=1)
            attribution = [t.numpy() for t in attribution]
            attribution = np.array(attribution)
            attribution = attribution.reshape(1, errors.shape[1])
            attribution = pd.DataFrame(attribution)
            attributions.append(attribution)

        attributions = pd.concat(attributions, axis=1)
        attributions = np.array(attributions)
        attributions = attributions.reshape(-1, errors.shape[1])
        attributions = pd.DataFrame(attributions)

        return attributions.abs()