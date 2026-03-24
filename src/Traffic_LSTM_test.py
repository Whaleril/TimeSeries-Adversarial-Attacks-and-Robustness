import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils import traffic_data, fgsm, bim, ktsa_fgsm, ktsa_bim, perturbation_ratio, rmse
from model import setup_lstm_model


model_path = "../Trained models/Traffic_regression_LSTM.h5"
epsilon = 0.2
alpha = 0.001
iterations = 200
ktsa_top_ratio = 0.1


train_X, train_y, val_X, val_y, test_X, test_y, scaler = traffic_data()


def invert_speed_predictions(predictions, test_context, scaler):
    inv_pred = np.concatenate((test_context, predictions), axis=1)
    inv_pred = scaler.inverse_transform(inv_pred)
    return inv_pred[:, 2]


def summarize_metrics(name, actual, predicted):
    rmse_value = np.sqrt(mean_squared_error(actual, predicted))
    mae_value = mean_absolute_error(actual, predicted)
    print("%s RMSE: %.3f" % (name, rmse_value))
    print("%s MAE: %.3f" % (name, mae_value))
    return {"rmse": rmse_value, "mae": mae_value}


if os.path.isfile(model_path):
    model = setup_lstm_model(train_X, 100)
    model.load_weights(model_path)

    adv_X, _ = fgsm(X=test_X, Y=test_y, model=model, loss_fn=rmse, epsilon=epsilon)
    bim_X, _ = bim(X=test_X, Y=test_y, model=model, loss_fn=rmse, epsilon=epsilon, alpha=alpha, I=iterations)
    ktsa_adv_X, _, ktsa_saliency, ktsa_mask = ktsa_fgsm(
        X=test_X,
        Y=test_y,
        model=model,
        loss_fn=rmse,
        epsilon=epsilon,
        top_ratio=ktsa_top_ratio,
    )
    ktsa_bim_X, _, _, _ = ktsa_bim(
        X=test_X,
        Y=test_y,
        model=model,
        loss_fn=rmse,
        epsilon=epsilon,
        alpha=alpha,
        I=iterations,
        top_ratio=ktsa_top_ratio,
    )

    adv_yhat = model.predict(adv_X)
    bim_yhat = model.predict(bim_X)
    ktsa_adv_yhat = model.predict(ktsa_adv_X)
    ktsa_bim_yhat = model.predict(ktsa_bim_X)
    yhat = model.predict(test_X)

    test_context = test_X[:, -1, :2]

    inv_adv_yhat = invert_speed_predictions(adv_yhat, test_context, scaler)
    inv_bim_yhat = invert_speed_predictions(bim_yhat, test_context, scaler)
    inv_ktsa_adv_yhat = invert_speed_predictions(ktsa_adv_yhat, test_context, scaler)
    inv_ktsa_bim_yhat = invert_speed_predictions(ktsa_bim_yhat, test_context, scaler)
    inv_yhat = invert_speed_predictions(yhat, test_context, scaler)

    test_y = test_y.reshape((len(test_y), 1))
    inv_y = invert_speed_predictions(test_y, test_context, scaler)

    clean_metrics = summarize_metrics("Clean", inv_y, inv_yhat)
    fgsm_metrics = summarize_metrics("FGSM", inv_y, inv_adv_yhat)
    bim_metrics = summarize_metrics("BIM", inv_y, inv_bim_yhat)
    ktsa_fgsm_metrics = summarize_metrics("KTSA-FGSM", inv_y, inv_ktsa_adv_yhat)
    ktsa_bim_metrics = summarize_metrics("KTSA-BIM", inv_y, inv_ktsa_bim_yhat)

    print("FGSM perturbation ratio: %.3f" % perturbation_ratio(test_X, adv_X))
    print("BIM perturbation ratio: %.3f" % perturbation_ratio(test_X, bim_X))
    print("KTSA-FGSM perturbation ratio: %.3f" % perturbation_ratio(test_X, ktsa_adv_X))
    print("KTSA-BIM perturbation ratio: %.3f" % perturbation_ratio(test_X, ktsa_bim_X))
    print("KTSA top ratio: %.3f" % ktsa_top_ratio)
    print("Average selected key-step ratio from mask: %.3f" % float(np.mean(ktsa_mask)))
    print("Average KTSA saliency score: %.6f" % float(np.mean(ktsa_saliency)))

    print("FGSM RMSE increase over Clean: %.3f" % (fgsm_metrics["rmse"] - clean_metrics["rmse"]))
    print("BIM RMSE increase over Clean: %.3f" % (bim_metrics["rmse"] - clean_metrics["rmse"]))
    print("KTSA-FGSM RMSE increase over Clean: %.3f" % (ktsa_fgsm_metrics["rmse"] - clean_metrics["rmse"]))
    print("KTSA-BIM RMSE increase over Clean: %.3f" % (ktsa_bim_metrics["rmse"] - clean_metrics["rmse"]))
    print("KTSA-FGSM RMSE delta vs FGSM: %.3f" % (ktsa_fgsm_metrics["rmse"] - fgsm_metrics["rmse"]))
    print("KTSA-BIM RMSE delta vs BIM: %.3f" % (ktsa_bim_metrics["rmse"] - bim_metrics["rmse"]))
    print("KTSA-FGSM MAE delta vs FGSM: %.3f" % (ktsa_fgsm_metrics["mae"] - fgsm_metrics["mae"]))
    print("KTSA-BIM MAE delta vs BIM: %.3f" % (ktsa_bim_metrics["mae"] - bim_metrics["mae"]))

    fig_verify = plt.figure(figsize=(20, 8))
    aa = [x for x in range(min(200, len(inv_y)))]
    plt.plot(aa, inv_y[: len(aa)], marker=".", label="actual")
    plt.plot(aa, inv_yhat[: len(aa)], "r", label="prediction")
    plt.plot(aa, inv_adv_yhat[: len(aa)], "g", label="fgsm prediction")
    plt.plot(aa, inv_bim_yhat[: len(aa)], "b", label="bim prediction")
    plt.plot(aa, inv_ktsa_adv_yhat[: len(aa)], "m", label="ktsa-fgsm prediction")
    plt.plot(aa, inv_ktsa_bim_yhat[: len(aa)], "c", label="ktsa-bim prediction")
    plt.ylabel("Speed", size=15)
    plt.xlabel("Time step", size=15)
    plt.legend(fontsize=15)
    fig_verify.savefig("../Images/Traffic_LSTM_test_plot.png")
    plt.close(fig_verify)
