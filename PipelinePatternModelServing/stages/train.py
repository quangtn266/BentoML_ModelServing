import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np
import os

df = pd.read_csv("/Users/quangtn/Desktop/01_work/01_job/02_ml/bentoml/"
                 "chapter9/stages/data/data.csv")

X = df['x'].to_numpy().reshape(-1, 1)
print(X)
y = df['y']
model = LogisticRegression(random_state=0).fit(X, y)

with open('/Users/quangtn/Desktop/01_work/01_job/02_ml/bentoml/'
          'chapter9/stages/model_location/model.pkl', 'wb') as f:
    pickle.dump(model, f)