import cppapi
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

neurons = 1
lr = 0.01
epochs = 10
eval_every = 10
batch_size = X_train.shape[0]
restart = True
verbose = 2

nn = cppapi.NeuralNetwork2([
        cppapi.Layer2(cppapi.LayerType.Dense, 16, cppapi.ActivationType.Sigmoid),
        cppapi.Layer2(cppapi.LayerType.Dense, 16, cppapi.ActivationType.Sigmoid),
        cppapi.Layer2(cppapi.LayerType.Dense, 1, cppapi.ActivationType.Linear)
    ], 
    cppapi.LossType.MSE)

trainer = cppapi.Trainer(nn, cppapi.OptimiserType.SGD, lr)

trainer.fit(X_train, Y_train, X_test, Y_test, epochs, eval_every, batch_size, restart, verbose)