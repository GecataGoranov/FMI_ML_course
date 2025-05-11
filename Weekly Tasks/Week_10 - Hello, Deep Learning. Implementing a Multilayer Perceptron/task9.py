import numpy as np
import matplotlib.pyplot as plt


def create_or_dataset():
    return [(0,0,0), (0,1,1), (1,0,1), (1,1,1)]


def create_and_dataset():
    return [(0,0,0), (0,1,0), (1,0,0), (1,1,1)]


def create_nand_dataset():
    return [(0,0,1), (0,1,1), (1,0,1), (1,1,0)]


def initialize_weights(x, y):
    return np.array([np.random.uniform(low=x, high=y) for _ in range(2)] + [1])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def calculate_loss(W, dataset):
    vals = []

    for data in dataset:
        z = W[0] * data[0] + W[1] * data[1] + W[2]
        y_hat = sigmoid(z)
        vals.append(-data[2] * np.log(y_hat) - (1 - data[2]) * np.log(1 - y_hat))

    return np.sum(vals)


def approx_derivatives(W, dataset, eps):
    return np.array([(calculate_loss([W[0] + eps, W[1], W[2]], dataset) - calculate_loss(W, dataset)) / eps,
            (calculate_loss([W[0], W[1] + eps, W[2]], dataset) - calculate_loss(W, dataset)) / eps,
            (calculate_loss([W[0], W[1], W[2] + eps], dataset) - calculate_loss(W, dataset)) / eps])


def fit(W, dataset, lr, epochs):
    loss = []
    for _ in range(epochs):
        W = W - approx_derivatives(W, dataset, 0.001) * lr
        loss.append(calculate_loss(W, dataset))

    return W, loss


def main():
    W_nand = initialize_weights(-10, 10)
    nand_data = create_nand_dataset()

    W_nand, nand_loss = fit(W_nand, nand_data, 0.001, 100000)
    print("NAND gate weights:", W_nand)
    print("NAND gate loss:", nand_loss[-1])

    plt.plot(nand_loss)
    plt.title("OR gate loss")
    plt.xlabel("Epochs")
    plt.ylabel("Log Loss")

    plt.tight_layout()
    plt.show()

    """
    Well, with NAND being the opposite of AND, the model just flipped the signs of its parameters.
    """


if __name__ == "__main__":
    main()