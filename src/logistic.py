import numpy as np

def sigmoid(z):
    # Clip the exponent input to reduce overflow risk.
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def hypothesis(X, theta):
    return sigmoid(X @ theta)

def cost(X, y, theta):
    h = np.clip(hypothesis(X, theta), 1e-12, 1 - 1e-12)
    y = np.asarray(y)
    return -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))

def fit_logistic(X, y, alpha, n_iters):
    n, d = X.shape
    theta = np.zeros(d)
    history = []
    y = np.asarray(y, dtype=float)
    for _ in range(n_iters):
        h = hypothesis(X, theta)
        gradient = (X.T @ (y - h)) / n
        theta = theta + alpha * gradient
        history.append(cost(X, y, theta))
    return theta, np.asarray(history)

def predict(X, theta):
    return (hypothesis(X, theta) >= 0.5).astype(int)
