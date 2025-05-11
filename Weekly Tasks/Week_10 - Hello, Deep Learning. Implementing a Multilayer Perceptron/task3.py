import numpy as np


def create_dataset(n):
    return [(i, i * 2) for i in range(n + 1)]


def initialize_weights(x, y):
    return np.random.uniform(low=x, high=y)

def calculate_loss(w, dataset):
    return (np.sum([np.square(data[1] - w * data[0]) for data in dataset]) / len(dataset))

def approx_derivative(w, dataset, eps):
    return (calculate_loss(w + eps, dataset) - calculate_loss(w, dataset)) / eps


def main():
    np.random.seed(42)
    w = initialize_weights(0, 10)

    dataset = create_dataset(6)

    loss = calculate_loss(w, dataset)
    print(f'MSE: {loss}')

    lr = 0.001

    derivative = approx_derivative(w, dataset, 0.001)
    new_loss = calculate_loss(w - derivative, dataset)
    print(f"New MSE without learning rate: {new_loss}")

    derivative = approx_derivative(w, dataset, 0.001)
    new_loss = calculate_loss(w - derivative * lr, dataset)
    print(f"New MSE with learning rate: {new_loss}")

    for i in range(10):
        w = w - approx_derivative(w, dataset, 0.001) * lr
        new_loss = calculate_loss(w, dataset)

    print(f"MSE after 10 epochs: {new_loss}")


if __name__ == "__main__":
    main()