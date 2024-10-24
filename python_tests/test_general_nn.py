import cppapi
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def test_trainer():

    diabetes = load_diabetes()
    targets = diabetes.target.astype(np.float32)
    data = diabetes.data.astype(np.float32)

    # Normalise the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    targets_ = targets.reshape(-1, 1)

    # make a train test split
    X_train, X_test, Y_train, Y_test = \
        train_test_split(data_scaled, targets_, test_size=0.2, random_state=42)

    sigmoid = cppapi.Sigmoid()
    neurons = 1
    lr = 0.01

    nn = cppapi.NeuralNetwork([
            cppapi.Dense(neurons, sigmoid)
        ], 
        cppapi.MeanSquareError())

    trainer = cppapi.Trainer(nn, cppapi.SGD(lr))

    epochs = 100
    eval_every = 10
    batch_size = 32
    restart = True
    verbose = 0

    trainer.fit(X_train, Y_train, X_test, Y_test, epochs, eval_every, batch_size, restart, verbose) 
    