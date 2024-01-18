import os

if not os.path.exists("/Users/quangtn/Desktop/01_work/01_job/"
                      "02_ml/bentoml/chapter9/stages/data"):
    os.mkdir("/Users/quangtn/Desktop/01_work/01_job/"
             "02_ml/bentoml/chapter9/stages/data")
    print("The 'data' directory is created")

if not os.path.exists("/Users/quangtn/Desktop/01_work/01_job/"
                      "02_ml/bentoml/chapter9/stages/model_location"):
    os.mkdir("/Users/quangtn/Desktop/01_work/01_job/"
                      "02_ml/bentoml/chapter9/stages/model_location")
    print("The directory 'model_location' is created")
