import cppapi
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.special import softmax
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

    trainer.fit(X_train, Y_train, X_test, Y_test, epochs, 
                eval_every, batch_size, restart, verbose)
    
def test_softmax_fn():
    from scipy.special import softmax
    
    A = np.random.randn(5, 3).astype(np.float32)
    softmax_np = softmax(A, axis=1)
    softmax_cpp = cppapi.loss_functions.softmax(A)
    atol = 1e-3
    rtol = 1e-3
    assert(np.allclose(softmax_np, softmax_cpp, rtol, atol))
    
def test_cross_entropy():
    n = 100
    X = np.random.randint(0, 3, (n, 1)).astype(np.float32)
    enc = OneHotEncoder(sparse_output=False)
    Xenc = enc.fit_transform(X)
    Q = np.random.randn(n, 3).astype(np.float32)
    pyPreds = softmax(Q, axis=1)
    loss = cppapi.LossFn(cppapi.LossType.CROSS_ENTROPY)
    cppapi.loss_functions.forward(loss, Q, Xenc.astype(np.float32))
    atol = 1e-3
    rtol = 1e-3
    assert(np.allclose(loss.place_holder, pyPreds))
    
    lossValue = np.sum(-Xenc * np.log(pyPreds) - (1-Xenc) * np.log(1-pyPreds)) / n
    
    assert(np.isclose(lossValue, loss.loss_value))

    
    