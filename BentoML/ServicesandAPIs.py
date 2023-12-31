import bentoml
import numpy as np
from bentoml.io import NumpyNdarray
regr_runner = bentoml.sklearn.get("dummyregressionmodel:5xvr5vvoysehz6xv").to_runner()
print(regr_runner)
service = bentoml.Service("DummyRegressionService", runners=[regr_runner])

@service.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict(input: np.ndarray) -> np.ndarray:
    print("input is ", input)
    response = regr_runner.run(input)
    print("Response is ", response)
    return response