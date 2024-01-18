import pandas as pd

df1 = pd.read_csv("/Users/quangtn/Desktop/01_work/01_job/"
           "02_ml/bentoml/chapter9/stages/data/data1.csv")

df2 = pd.read_csv("/Users/quangtn/Desktop/01_work/01_job/"
           "02_ml/bentoml/chapter9/stages/data/data2.csv")

df = pd.concat([df1, df2], ignore_index=True)

print(df)

df.to_csv("/Users/quangtn/Desktop/01_work/01_job/"
           "02_ml/bentoml/chapter9/stages/data/data.csv")