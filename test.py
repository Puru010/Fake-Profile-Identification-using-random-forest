import pandas as pd
import joblib

model = joblib.load('./data/rf_model.pkl')
input_data = pd.read_csv('./test-data/values.csv')
predictions = model.predict(input_data)
label_mapping = {0: 'Real', 1: 'Fake'}
predicted_labels = [label_mapping[pred] for pred in predictions]

for idx, label in enumerate(predicted_labels):
    print(f"Profile {idx + 1}: {label}")
