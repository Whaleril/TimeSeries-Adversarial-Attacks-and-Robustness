import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from utils import traffic_data, fgsm, bim, rmse
from model import setup_lstm_model


model_path = "../Trained models/Traffic_regression_LSTM.h5"


train_X, train_y, val_X, val_y, test_X, test_y, scaler = traffic_data()


if os.path.isfile(model_path):
    model = setup_lstm_model(train_X, 100)
    model.load_weights(model_path)

    adv_X, _ = fgsm(X=test_X, Y=test_y, model=model, loss_fn=rmse, epsilon=0.2)
    bim_X, _ = bim(X=test_X, Y=test_y, model=model, loss_fn=rmse, epsilon=0.2, alpha=0.001, I=200)

    adv_yhat = model.predict(adv_X)
    bim_yhat = model.predict(bim_X)
    yhat = model.predict(test_X)

    test_context = test_X[:, -1, :2]

    inv_adv_yhat = np.concatenate((test_context, adv_yhat), axis=1)
    inv_adv_yhat = scaler.inverse_transform(inv_adv_yhat)
    inv_adv_yhat = inv_adv_yhat[:, 2]

    inv_bim_yhat = np.concatenate((test_context, bim_yhat), axis=1)
    inv_bim_yhat = scaler.inverse_transform(inv_bim_yhat)
    inv_bim_yhat = inv_bim_yhat[:, 2]

    inv_yhat = np.concatenate((test_context, yhat), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 2]

    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_context, test_y), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 2]

    test_rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print("Test RMSE: %.3f" % test_rmse)

    fgsm_rmse = np.sqrt(mean_squared_error(inv_y, inv_adv_yhat))
    print("Test adv RMSE: %.3f" % fgsm_rmse)

    bim_rmse = np.sqrt(mean_squared_error(inv_y, inv_bim_yhat))
    print("Test BIM RMSE: %.3f" % bim_rmse)

    fig_verify = plt.figure(figsize=(20, 8))
    aa = [x for x in range(min(200, len(inv_y)))]
    plt.plot(aa, inv_y[: len(aa)], marker=".", label="actual")
    plt.plot(aa, inv_yhat[: len(aa)], "r", label="prediction")
    plt.plot(aa, inv_adv_yhat[: len(aa)], "g", label="adv prediction")
    plt.plot(aa, inv_bim_yhat[: len(aa)], "b", label="bim prediction")
    plt.ylabel("Speed", size=15)
    plt.xlabel("Time step", size=15)
    plt.legend(fontsize=15)
    fig_verify.savefig("../Images/Traffic_LSTM_test_plot.png")
    plt.close(fig_verify)
