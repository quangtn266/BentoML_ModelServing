import numpy as np
import random

def predict_phase_on_model(x):
    print("The value of x is", x)
    if x < 0.5:
        return True
    else:
        return False

def predict_phase_two_models():
    print("Phase 2 model is called")
    prediction = np.random.choice(["ClassA", "ClassB", "ClassC"])
    return prediction

if __name__ == "__main__":
    phase_one_prediction = predict_phase_on_model(random.uniform(0, 1))

    if phase_one_prediction == True:
        response = predict_phase_two_models()
        print(response)
    else:
        print("Phase 2 models is not called")