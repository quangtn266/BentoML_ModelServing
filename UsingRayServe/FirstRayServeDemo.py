import ray
from ray import serve

@serve.deployment
class MyFirstDeployment:
    # Take the message to return as an argument to the constructor.
    def __init__(self, msg):
        self.msg = msg

    def __call__(self):
        return self.msg

my_first_deployment = MyFirstDeployment.bind("Hello World")

##### RUNING: serve run FirstRayServeDemo:my_first_deployment

#handle = serve.run(my_first_deployment)
#print(ray.get(handle.remote())) # Hello World !!!
