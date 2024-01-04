import pandas as pd

df1 = pd.DataFrame({"x": [5, 5, 5, 3, 4, 5], "y": [2, 2, 2, 1, 1, 1]})

df1.to_csv("/Users/quangtn/Desktop/01_work/01_job/"
           "02_ml/bentoml/chapter9/stages/data/data2.csv")