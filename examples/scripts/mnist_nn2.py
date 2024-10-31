import cppapi
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.special import softmax
import numpy as np

digits = load_digits()

n_samples = len(digits.images) 
print(n_samples)

data = digits.images.reshape((n_samples, -1))
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.3, shuffle=False
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

encoder = OneHotEncoder(sparse_output=False)
y_train_enc = encoder.fit_transform(y_train).astype(np.float32)
y_test_enc = encoder.fit_transform(y_test).astype(np.float32)

# We also have to apply one hot encoding to the target data


print("train data:", X_train.shape, "train target:", y_train_enc.shape, 
      "test data:", X_test.shape, "target data:", y_test_enc.shape)

lr = 0.1
epochs = 50
eval_every = 10
batch_size = 60
restart = True
verbose = 2

nn = cppapi.NeuralNetwork2([
        cppapi.Layer2(cppapi.LayerType.Dense, 89, cppapi.ActivationType.Tanh),
        cppapi.Layer2(cppapi.LayerType.Dense, 10, cppapi.ActivationType.Sigmoid),
    ], cppapi.LossType.CROSS_ENTROPY)

trainer = cppapi.Trainer(nn, cppapi.OptimiserType.SGD, lr)
trainer.fit(X_train, y_train_enc, X_test, y_test_enc, epochs,
            eval_every, batch_size, restart, verbose)

logit_predictions = nn.predict(X_test)
probs = np.clip(softmax(logit_predictions), 1e-9, 1 - 1e-9)
print(logit_predictions)
print(probs.shape)
predictions = np.argmax(probs, axis=1)
print(predictions)
print(y_test.reshape(y_test.shape[0],))
acc = np.sum(np.where(predictions == y_test.reshape(y_test.shape[0], ), 1, 0))\
    / predictions.shape[0]
print(f"Accuracy: {acc * 100:.3}%")