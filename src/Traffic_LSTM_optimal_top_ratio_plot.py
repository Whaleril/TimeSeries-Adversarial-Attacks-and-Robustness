import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils import traffic_data, fgsm, bim, ktsa_fgsm, ktsa_bim, perturbation_ratio, rmse
from model import setup_lstm_model


model_path = "../Trained models/Traffic_regression_LSTM.h5"
output_path = "../Images/Traffic_LSTM_test_plot.png"

# 攻击参数设置
epsilon = 0.2
alpha = 0.001
iterations = 200

# 在最终绘图前先扫描的候选 top ratio
candidate_top_ratios = [0.05, 0.10, 0.20, 0.30, 0.50]

# 用于选择最优 top ratio 的综合评分权重
# 这里略微偏向攻击有效性，但仍然保留对稀疏性的奖励
effectiveness_weight = 0.65
sparsity_weight = 0.35

# 为了便于观察，只绘制前 N 个测试点
plot_points = 200


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


def normalize_scores(values):
    values = np.array(values, dtype=np.float64)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value - min_value < 1e-12:
        return np.ones_like(values)
    return (values - min_value) / (max_value - min_value)


def weighted_harmonic_score(effectiveness_score, sparsity_score):
    numerator = effectiveness_weight + sparsity_weight
    denominator = (effectiveness_weight / max(effectiveness_score, 1e-12)) + (
        sparsity_weight / max(sparsity_score, 1e-12)
    )
    return numerator / denominator


def select_best_ratio(scan_rows, method_name):
    effectiveness_values = [row["rmse_increase"] for row in scan_rows]
    sparsity_values = [1.0 - row["perturbation_ratio"] for row in scan_rows]

    effectiveness_scores = normalize_scores(effectiveness_values)
    sparsity_scores = normalize_scores(sparsity_values)

    best_row = None
    best_score = -1.0

    print("\n%s ratio scan summary:" % method_name)
    for index, row in enumerate(scan_rows):
        row["effectiveness_score"] = float(effectiveness_scores[index])
        row["sparsity_score"] = float(sparsity_scores[index])
        row["composite_score"] = float(
            weighted_harmonic_score(row["effectiveness_score"], row["sparsity_score"])
        )

        print(
            "ratio=%.2f | rmse_inc=%.3f | perturb_ratio=%.3f | eff_score=%.3f | sparse_score=%.3f | composite=%.3f"
            % (
                row["top_ratio"],
                row["rmse_increase"],
                row["perturbation_ratio"],
                row["effectiveness_score"],
                row["sparsity_score"],
                row["composite_score"],
            )
        )

        # 平分时的判定规则：如果综合分数非常接近，则优先选择更稀疏的攻击
        if row["composite_score"] > best_score + 1e-12:
            best_row = row
            best_score = row["composite_score"]
        elif abs(row["composite_score"] - best_score) <= 1e-6:
            if row["perturbation_ratio"] < best_row["perturbation_ratio"] - 1e-12:
                best_row = row
            elif abs(row["perturbation_ratio"] - best_row["perturbation_ratio"]) <= 1e-6:
                if row["rmse_increase"] > best_row["rmse_increase"] + 1e-12:
                    best_row = row

    print(
        "Selected best %s top ratio: %.2f (composite=%.3f)"
        % (method_name, best_row["top_ratio"], best_row["composite_score"])
    )
    return best_row["top_ratio"], best_row


train_X, train_y, val_X, val_y, test_X, test_y, scaler = traffic_data()


if os.path.isfile(model_path):
    model = setup_lstm_model(train_X, 100)
    model.load_weights(model_path)

    clean_yhat = model.predict(test_X, verbose=0)

    fgsm_X, _ = fgsm(X=test_X, Y=test_y, model=model, loss_fn=rmse, epsilon=epsilon)
    bim_X, _ = bim(
        X=test_X,
        Y=test_y,
        model=model,
        loss_fn=rmse,
        epsilon=epsilon,
        alpha=alpha,
        I=iterations,
    )

    fgsm_yhat = model.predict(fgsm_X, verbose=0)
    bim_yhat = model.predict(bim_X, verbose=0)

    test_context = test_X[:, -1, :2]

    inv_clean_yhat = invert_speed_predictions(clean_yhat, test_context, scaler)
    inv_fgsm_yhat = invert_speed_predictions(fgsm_yhat, test_context, scaler)
    inv_bim_yhat = invert_speed_predictions(bim_yhat, test_context, scaler)

    test_y = test_y.reshape((len(test_y), 1))
    inv_y = invert_speed_predictions(test_y, test_context, scaler)

    clean_metrics = summarize_metrics("Clean", inv_y, inv_clean_yhat)
    fgsm_metrics = summarize_metrics("FGSM", inv_y, inv_fgsm_yhat)
    bim_metrics = summarize_metrics("BIM", inv_y, inv_bim_yhat)

    ktsa_fgsm_scan_rows = []
    ktsa_bim_scan_rows = []

    print("\n开始扫描候选 top ratio，并在最终绘图前自动选择最优比例...")
    for ratio in candidate_top_ratios:
        scanned_ktsa_fgsm_X, _, _, _ = ktsa_fgsm(
            X=test_X,
            Y=test_y,
            model=model,
            loss_fn=rmse,
            epsilon=epsilon,
            top_ratio=ratio,
        )
        scanned_ktsa_bim_X, _, _, _ = ktsa_bim(
            X=test_X,
            Y=test_y,
            model=model,
            loss_fn=rmse,
            epsilon=epsilon,
            alpha=alpha,
            I=iterations,
            top_ratio=ratio,
        )

        scanned_ktsa_fgsm_yhat = model.predict(scanned_ktsa_fgsm_X, verbose=0)
        scanned_ktsa_bim_yhat = model.predict(scanned_ktsa_bim_X, verbose=0)

        inv_scanned_ktsa_fgsm = invert_speed_predictions(scanned_ktsa_fgsm_yhat, test_context, scaler)
        inv_scanned_ktsa_bim = invert_speed_predictions(scanned_ktsa_bim_yhat, test_context, scaler)

        scanned_ktsa_fgsm_rmse = np.sqrt(mean_squared_error(inv_y, inv_scanned_ktsa_fgsm))
        scanned_ktsa_bim_rmse = np.sqrt(mean_squared_error(inv_y, inv_scanned_ktsa_bim))

        ktsa_fgsm_scan_rows.append(
            {
                "top_ratio": ratio,
                "rmse_increase": float(scanned_ktsa_fgsm_rmse - clean_metrics["rmse"]),
                "perturbation_ratio": float(perturbation_ratio(test_X, scanned_ktsa_fgsm_X)),
            }
        )
        ktsa_bim_scan_rows.append(
            {
                "top_ratio": ratio,
                "rmse_increase": float(scanned_ktsa_bim_rmse - clean_metrics["rmse"]),
                "perturbation_ratio": float(perturbation_ratio(test_X, scanned_ktsa_bim_X)),
            }
        )

    best_top_ratio_fgsm, best_fgsm_row = select_best_ratio(ktsa_fgsm_scan_rows, "KTSA-FGSM")
    best_top_ratio_bim, best_bim_row = select_best_ratio(ktsa_bim_scan_rows, "KTSA-BIM")

    ktsa_fgsm_X, _, ktsa_fgsm_saliency, ktsa_fgsm_mask = ktsa_fgsm(
        X=test_X,
        Y=test_y,
        model=model,
        loss_fn=rmse,
        epsilon=epsilon,
        top_ratio=best_top_ratio_fgsm,
    )
    ktsa_bim_X, _, ktsa_bim_saliency, ktsa_bim_mask = ktsa_bim(
        X=test_X,
        Y=test_y,
        model=model,
        loss_fn=rmse,
        epsilon=epsilon,
        alpha=alpha,
        I=iterations,
        top_ratio=best_top_ratio_bim,
    )

    ktsa_fgsm_yhat = model.predict(ktsa_fgsm_X, verbose=0)
    ktsa_bim_yhat = model.predict(ktsa_bim_X, verbose=0)

    inv_ktsa_fgsm_yhat = invert_speed_predictions(ktsa_fgsm_yhat, test_context, scaler)
    inv_ktsa_bim_yhat = invert_speed_predictions(ktsa_bim_yhat, test_context, scaler)

    ktsa_fgsm_metrics = summarize_metrics("KTSA-FGSM", inv_y, inv_ktsa_fgsm_yhat)
    ktsa_bim_metrics = summarize_metrics("KTSA-BIM", inv_y, inv_ktsa_bim_yhat)

    print("FGSM perturbation ratio: %.3f" % perturbation_ratio(test_X, fgsm_X))
    print("BIM perturbation ratio: %.3f" % perturbation_ratio(test_X, bim_X))
    print("KTSA-FGSM perturbation ratio: %.3f" % perturbation_ratio(test_X, ktsa_fgsm_X))
    print("KTSA-BIM perturbation ratio: %.3f" % perturbation_ratio(test_X, ktsa_bim_X))
    print("Selected KTSA-FGSM top ratio: %.3f" % best_top_ratio_fgsm)
    print("Selected KTSA-BIM top ratio: %.3f" % best_top_ratio_bim)
    print("Selected KTSA-FGSM composite score: %.3f" % best_fgsm_row["composite_score"])
    print("Selected KTSA-BIM composite score: %.3f" % best_bim_row["composite_score"])
    print("Average KTSA-FGSM selected ratio from mask: %.3f" % float(np.mean(ktsa_fgsm_mask)))
    print("Average KTSA-BIM selected ratio from mask: %.3f" % float(np.mean(ktsa_bim_mask)))
    print("Average KTSA-FGSM saliency score: %.6f" % float(np.mean(ktsa_fgsm_saliency)))
    print("Average KTSA-BIM saliency score: %.6f" % float(np.mean(ktsa_bim_saliency)))

    print("FGSM RMSE increase over Clean: %.3f" % (fgsm_metrics["rmse"] - clean_metrics["rmse"]))
    print("BIM RMSE increase over Clean: %.3f" % (bim_metrics["rmse"] - clean_metrics["rmse"]))
    print("KTSA-FGSM RMSE increase over Clean: %.3f" % (ktsa_fgsm_metrics["rmse"] - clean_metrics["rmse"]))
    print("KTSA-BIM RMSE increase over Clean: %.3f" % (ktsa_bim_metrics["rmse"] - clean_metrics["rmse"]))
    print("KTSA-FGSM RMSE delta vs FGSM: %.3f" % (ktsa_fgsm_metrics["rmse"] - fgsm_metrics["rmse"]))
    print("KTSA-BIM RMSE delta vs BIM: %.3f" % (ktsa_bim_metrics["rmse"] - bim_metrics["rmse"]))

    fig_verify = plt.figure(figsize=(20, 8))
    aa = [x for x in range(min(plot_points, len(inv_y)))]
    plt.plot(aa, inv_y[: len(aa)], marker=".", linewidth=1.5, label="Actual")
    plt.plot(aa, inv_clean_yhat[: len(aa)], "r", linewidth=2, label="Clean")
    plt.plot(aa, inv_fgsm_yhat[: len(aa)], "g", linewidth=2, label="FGSM")
    plt.plot(aa, inv_bim_yhat[: len(aa)], "b", linewidth=2, label="BIM")
    plt.plot(
        aa,
        inv_ktsa_fgsm_yhat[: len(aa)],
        "m",
        linewidth=2,
        label="KTSA-FGSM (top_ratio=%.2f)" % best_top_ratio_fgsm,
    )
    plt.plot(
        aa,
        inv_ktsa_bim_yhat[: len(aa)],
        "c",
        linewidth=2,
        label="KTSA-BIM (top_ratio=%.2f)" % best_top_ratio_bim,
    )
    plt.ylabel("Speed", size=15)
    plt.xlabel("Time step", size=15)
    plt.title("Traffic LSTM Prediction Curves Under Standard and KTSA Attacks", size=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    fig_verify.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig_verify)

    print("Plot saved to: %s" % output_path)
else:
    print("Model file not found: %s" % model_path)
