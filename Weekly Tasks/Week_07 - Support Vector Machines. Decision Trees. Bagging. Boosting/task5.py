import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def log_loss(raw_model_output):
    return np.log(1 + np.exp(-raw_model_output))

def hinge_loss(raw_model_output):
    return np.max([0.0, 1.0 - raw_model_output])


def main():
    outputs = np.linspace(-2, 2, 1000)
    logistic_outputs = np.array([log_loss(output) for output in outputs])
    print(logistic_outputs)
    hinge_outputs = np.array([hinge_loss(output) for output in outputs])

    plt.plot(outputs, logistic_outputs)
    plt.plot(outputs, hinge_outputs)
    plt.show()


if __name__ == "__main__":
    main()