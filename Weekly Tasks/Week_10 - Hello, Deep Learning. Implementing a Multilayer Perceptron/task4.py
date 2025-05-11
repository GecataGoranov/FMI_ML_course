import numpy as np


def create_dataset(n):
    return [(i, i * 2) for i in range(n + 1)]


def initialize_weights(x, y):
    return np.random.uniform(low=x, high=y)

def calculate_loss(w, dataset):
    return (np.sum([np.square(data[1] - w * data[0]) for data in dataset]) / len(dataset))

def approx_derivative(w, dataset, eps):
    return (calculate_loss(w + eps, dataset) - calculate_loss(w, dataset)) / eps

def fit(w, dataset, lr, epochs):
    loss = 0
    for i in range(epochs):
        w = w - approx_derivative(w, dataset, 0.001) * lr
        loss = calculate_loss(w, dataset)
        print(loss)

    return w
    
def main():
    np.random.seed(42)
    w = initialize_weights(0, 10)

    dataset = create_dataset(6)

    print("W: ", w)
    w = fit(w, dataset, 0.001, 500)
    print("W: ", w)


if __name__ == "__main__":
    main()