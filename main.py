import pandas as pd
from joblib import load
model, x_train = load('diagnosis.joblib')

def make_prediction(model, feature_names):
  take_input = dict()
  for f in feature_names:
    value = input(f'{f}[0: No/1: Yes]: ')
    take_input[f] = int(value)
  input_df = pd.DataFrame([take_input])
#   Utile pour le web scraping
  feature_true = [k for k,v in take_input.items() if v == 1 ]
  prediction = model.predict(input_df)
  return prediction, feature_true

feature_names = x_train.columns
prediction, feature_true = make_prediction(model, feature_names)
print(f'Predicted prognosis: {prediction[0]}')
print(f'Predicted prognosis: {feature_true}')