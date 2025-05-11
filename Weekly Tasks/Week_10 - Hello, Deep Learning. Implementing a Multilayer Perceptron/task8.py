import numpy as np
import matplotlib.pyplot as plt


def create_or_dataset():
    return [(0,0,0), (0,1,1), (1,0,1), (1,1,1)]


def create_and_dataset():
    return [(0,0,0), (0,1,0), (1,0,0), (1,1,1)]


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
    for i in range(epochs):
        W = W - approx_derivatives(W, dataset, 0.001) * lr
        loss.append(calculate_loss(W, dataset))

    return W, loss


def main():
    W_or = initialize_weights(-10, 10)
    or_data = create_or_dataset()

    W_and = initialize_weights(-10, 10)
    and_data = create_and_dataset()

    W_or, or_loss = fit(W_or, or_data, 0.001, 100000)
    print("OR gate weights:", W_or)
    print("OR gate loss:", or_loss[-1])

    W_and, and_loss = fit(W_and, and_data, 0.001, 100000)
    print("AND gate weights", W_and)
    print("AND gate loss:", and_loss[-1])

    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)

    axes[0].plot(or_loss)
    axes[0].set_title("OR gate loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Log Loss")

    axes[1].plot(and_loss)
    axes[1].set_title("AND gate loss")
    axes[1].set_xlabel("Epochs")

    plt.tight_layout()
    plt.show()

    """
    The sigmoid makes the model even more confident, as it squishes the results between 0 and 1.
    """


if __name__ == "__main__":
    main()