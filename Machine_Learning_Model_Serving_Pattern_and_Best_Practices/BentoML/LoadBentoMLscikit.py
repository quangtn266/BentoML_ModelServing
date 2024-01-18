import bentoml
from sklearn.ensemble import RandomForestRegressor
regr: RandomForestRegressor = bentoml.sklearn.load_model("dummyregressionmodel:5xvr5vvoysehz6xv")
print(regr)