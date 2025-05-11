import numpy as np


def create_xor_data():
    return [(0,0,0), (0,1,1), (1,0,1), (1,1,0)]


class Xor:
    def __init__(self, eps=0.001, lr=0.001, epochs=100000):
        self.dataset = []
        self.W1 = np.array([np.random.uniform(-1, 1) for _ in range(2)] + [1])
        self.W2 = np.array([np.random.uniform(-1, 1) for _ in range(2)] + [1])
        self.W3 = np.array([np.random.uniform(-1, 1) for _ in range(2)] + [1])

        self.eps = eps
        self.lr = lr
        self.epochs = epochs

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _calculate_loss(self, W):
        vals = []

        W1, W2, W3 = W

        for data in self.dataset:
            Z1 = self._sigmoid(W1[0] * data[0] + W1[1] * data[1] + W1[2])
            Z2 = self._sigmoid(W2[0] * data[0] + W2[1] * data[1] + W2[2])
            Z3 = self._sigmoid(W3[0] * Z1 + W3[1] * Z2 + W3[2])

            vals.append(-data[2] * np.log(Z3) - (1 - data[2]) * np.log(1 - Z3))

        return np.sum(vals)
    
    def _loss_with_eps(self, W, i, j):
        W_new = W.copy()
        W_new[i][j] = W_new[i][j] + self.eps
        return self._calculate_loss(W_new)

    def _approx_derivatives(self):
        derivatives = []
        W = np.array([self.W1, self.W2, self.W3])
        for i in range(3):
            for j in range(3):
                derivatives.append(self._loss_with_eps(W, i, j))

        return np.array(derivatives)
    
    def fit(self, data):
        self.dataset = data
        loss = []
        for _ in range(self.epochs):

            derivatives = self._approx_derivatives()
            self.W1 = self.W1 - derivatives[:3] * self.lr
            self.W2 = self.W2 - derivatives[3:6] * self.lr
            self.W3 = self.W3 - derivatives[6:] * self.lr
            loss.append(self._calculate_loss([self.W1, self.W2, self.W3]))

        return [self.W1, self.W2, self.W3], loss
    
    def predict(self, x1, x2):
        data = [x1, x2]
        Z1 = self._sigmoid(self.W1[0] * data[0] + self.W1[1] * data[1] + self.W1[2])
        Z2 = self._sigmoid(self.W2[0] * data[0] + self.W2[1] * data[1] + self.W2[2])
        Z3 = self._sigmoid(self.W3[0] * Z1 + self.W3[1] * Z2 + self.W3[2])

        return Z3


def forward(model, x1, x2):
    return model.predict(x1, x2)


def main():
    xor_model = Xor()
    dataset = create_xor_data()
    W, loss = xor_model.fit(dataset)

    print("W: ", W)
    print("Loss:", loss[-1])

    print("Prediction for (0,0):", forward(xor_model, 0, 0))
    print("Prediction for (0,1):", forward(xor_model, 0, 1))
    print("Prediction for (1,0):", forward(xor_model, 1, 0))
    print("Prediction for (1,1):", forward(xor_model, 1, 1))

    """
    Something is really wrong here.
    Maybe something with the math.
    """


if __name__ == "__main__":
    main()