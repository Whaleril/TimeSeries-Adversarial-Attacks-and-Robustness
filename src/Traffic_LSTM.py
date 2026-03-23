import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from utils import traffic_data
from model import setup_lstm_model


model_path = "../Trained models/Traffic_regression_LSTM.h5"


train_X, train_y, val_X, val_y, test_X, test_y, scaler = traffic_data()

model = setup_lstm_model(train_X, 100)
history = model.fit(
    train_X,
    train_y,
    epochs=200,
    batch_size=32,
    validation_data=(val_X, val_y),
    verbose=2,
)

model.save(model_path)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "validation"], loc="upper right")
plt.savefig("../Images/Traffic_LSTM_train_loss.png")
plt.close()

yhat = model.predict(test_X)

test_context = test_X[:, -1, :2]

inv_yhat = np.concatenate((test_context, yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 2]

test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_context, test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 2]

test_rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print("Test RMSE: %.3f" % test_rmse)

fig_verify = plt.figure(figsize=(20, 8))
aa = [x for x in range(min(200, len(inv_y)))]
plt.plot(aa, inv_y[: len(aa)], marker=".", label="actual")
plt.plot(aa, inv_yhat[: len(aa)], "r", label="prediction")
plt.ylabel("Speed", size=15)
plt.xlabel("Time step", size=15)
plt.legend(fontsize=15)
fig_verify.savefig("../Images/Traffic_LSTM_train_plot.png")
plt.close(fig_verify)
