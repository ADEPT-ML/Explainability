import pandas as pd
import tensorflow as tf

from PG_deep_models.src.dagmm_master.ExplainDagmmbyUsingSHAP import ExplainDagmmByUsingSHAP

# Path
feature_list = ["Strom Leistung", "Strom 2 Leistung", "Strom 3 Leistung"]
root = "E:\\ML\\PG_deep_models\\data\\"
data_file = "OH12.csv"

# Load Data
df = pd.read_csv(root + data_file)[feature_list]

# Load trained Dagmm model
model = tf.keras.models.load_model(root + "OH12_S3", compile=False)

# Initialize the explainer
explain_model = ExplainDagmmByUsingSHAP(model=model, data=df, anomaly_index=40711)

# Results
result = explain_model.explain()
print(result)
