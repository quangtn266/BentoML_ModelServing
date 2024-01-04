import pandas as pd

df1 = pd.DataFrame({"x": [2, 2, 2, 3, 4, 5], "y": [1, 1, 1, 1, 1, 1]})

df1.to_csv("/Users/quangtn/Desktop/01_work/01_job/"
           "02_ml/bentoml/chapter9/stages/data/data1.csv")