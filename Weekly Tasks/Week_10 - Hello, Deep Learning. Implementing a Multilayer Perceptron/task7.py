import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    x = np.linspace(-10, 10, 1000)
    plt.plot(x, sigmoid(x))
    plt.show()

    """
    I'm not really sure if that was what I was supposed to do in this task.
    """


if __name__ == "__main__":
    main()