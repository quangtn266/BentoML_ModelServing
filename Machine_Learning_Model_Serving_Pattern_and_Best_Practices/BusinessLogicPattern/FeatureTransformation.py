import pandas as pd

df = pd.DataFrame({"climate": ["Sunny", "Rainy", "Cloudy"]})
#print("Initial data")
#print(df.head())
df2 = pd.get_dummies(df)
#print("Data after one hot encoding")
#print(df2)


response = [0, 0, 0, 1, 0, 1, 0, 2, 3]
mapping = {0: "Rose", 1: "Sunflower", 2: "Marigold", 3: "Lotus"}
converter = lambda x: mapping[x]
final_response = [converter(x) for x in response]
print(final_response)