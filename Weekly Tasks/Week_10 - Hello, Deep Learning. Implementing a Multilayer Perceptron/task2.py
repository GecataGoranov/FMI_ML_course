import numpy as np

def create_dataset(n):
    return [(i, i * 2) for i in range(n)]


def initialize_weights(x, y):
    return np.random.uniform(low=x, high=y)

def calculate_loss(w, dataset):
    return (np.sum([np.square(data[1] - w * data[0]) for data in dataset]) / len(dataset))


def main():
    np.random.seed(42)
    w = initialize_weights(0, 10)

    dataset = create_dataset(6)

    loss = calculate_loss(w, dataset)
    print(f'MSE: {loss}')

    """
    While w decreases, the loss also decreases.
    """


if __name__ == "__main__":
    main()