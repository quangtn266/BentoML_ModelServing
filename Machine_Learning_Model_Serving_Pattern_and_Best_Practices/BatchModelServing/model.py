import pandas as pd
import random

random_scores = []
for i in range(0, 5):
    x = round(random.random(), 2)
    random_scores.append(x)

df = pd.DataFrame(
    {
        "products": ["Product A", "Product B", "Product C", "Product D", "Product E"],
        "scores": random_scores
    }
)

df.to_csv("/Users/quangtn/Desktop/01_work/01_job/02_ml/bentoml/chapter6/predictions.csv")