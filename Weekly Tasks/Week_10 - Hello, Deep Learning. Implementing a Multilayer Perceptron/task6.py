import numpy as np


def create_or_dataset():
    return [(0,0,0), (0,1,1), (1,0,1), (1,1,1)]

def create_and_dataset():
    return [(0,0,0), (0,1,0), (1,0,0), (1,1,1)]

def initialize_weights(x, y):
    return np.array([np.random.uniform(low=x, high=y) for _ in range(2)] + [1])

def calculate_loss(W, dataset):
    return np.sum([np.square(data[2] - (W[0] * data[0] + W[1] * data[1] + W[2])) for data in dataset])

def approx_derivatives(W, dataset, eps):
    return np.array([(calculate_loss([W[0] + eps, W[1], W[2]], dataset) - calculate_loss(W, dataset)) / eps,
            (calculate_loss([W[0], W[1] + eps, W[2]], dataset) - calculate_loss(W, dataset)) / eps,
            (calculate_loss([W[0], W[1], W[2] + eps], dataset) - calculate_loss(W, dataset)) / eps])

def fit(W, dataset, lr, epochs):
    loss = 0
    for i in range(epochs):
        W = W - approx_derivatives(W, dataset, 0.001) * lr
        loss = calculate_loss(W, dataset)

    return W, loss


def main():
    W_or = initialize_weights(-10, 10)
    or_data = create_or_dataset()

    W_and = initialize_weights(-10, 10)
    and_data = create_and_dataset()

    W_or, or_loss = fit(W_or, or_data, 0.001, 100000)
    print("OR gate weights:", W_or)
    print("OR gate loss:", or_loss)

    W_and, and_loss = fit(W_and, and_data, 0.001, 100000)
    print("AND gate weights", W_and)
    print("AND gate loss:", and_loss)

    """
    The results are closer to actually being 0 and 1 and therefore the loss has decreased.
    The model is becoming more confident.
    """


if __name__ == "__main__":
    main()