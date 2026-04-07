import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 导入所有需要的基线和刚才新写的 Random 攻击方法
from utils import traffic_data, ktsa_fgsm, ktsa_bim, random_fgsm, random_bim, rmse
from model import setup_lstm_model

# 设置随机种子，保证每次运行生成的随机时间步一致，方便复现
np.random.seed(42)

model_path = "../Trained models/Traffic_regression_LSTM.h5"
epsilon = 0.2
alpha = 0.001
iterations = 200

# 任务 4.1 需要扫描的关键时间步比例
top_ratios = [0.05, 0.10, 0.20, 0.30, 0.50]

train_X, train_y, val_X, val_y, test_X, test_y, scaler = traffic_data()


def invert_speed_predictions(predictions, test_context, scaler):
    inv_pred = np.concatenate((test_context, predictions), axis=1)
    inv_pred = scaler.inverse_transform(inv_pred)
    return inv_pred[:, 2]


if os.path.isfile(model_path):
    model = setup_lstm_model(train_X, 100)
    model.load_weights(model_path)

    test_context = test_X[:, -1, :2]

    # 1. 计算 Clean 状态下的预测结果和 RMSE
    yhat = model.predict(test_X, verbose=0)
    inv_yhat = invert_speed_predictions(yhat, test_context, scaler)
    test_y_reshaped = test_y.reshape((len(test_y), 1))
    inv_y = invert_speed_predictions(test_y_reshaped, test_context, scaler)
    clean_rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print(f"Clean RMSE: {clean_rmse:.3f}\n")

    # 用于保存画图数据的列表
    ktsa_fgsm_rmse_inc = []
    random_fgsm_rmse_inc = []
    ktsa_bim_rmse_inc = []
    random_bim_rmse_inc = []

    # 2. 遍历不同的 top_ratio
    for ratio in top_ratios:
        print(f"========== Testing top_ratio = {ratio:.2f} ==========")

        # 运行攻击
        ktsa_adv_X, _, _, _ = ktsa_fgsm(X=test_X, Y=test_y, model=model, loss_fn=rmse, epsilon=epsilon, top_ratio=ratio)
        random_adv_X, _, _ = random_fgsm(X=test_X, Y=test_y, model=model, loss_fn=rmse, epsilon=epsilon,
                                         top_ratio=ratio)

        ktsa_bim_X, _, _, _ = ktsa_bim(X=test_X, Y=test_y, model=model, loss_fn=rmse, epsilon=epsilon, alpha=alpha,
                                       I=iterations, top_ratio=ratio)
        random_bim_X, _, _ = random_bim(X=test_X, Y=test_y, model=model, loss_fn=rmse, epsilon=epsilon, alpha=alpha,
                                        I=iterations, top_ratio=ratio)

        # 预测并逆归一化
        inv_ktsa_fgsm = invert_speed_predictions(model.predict(ktsa_adv_X, verbose=0), test_context, scaler)
        inv_random_fgsm = invert_speed_predictions(model.predict(random_adv_X, verbose=0), test_context, scaler)
        inv_ktsa_bim = invert_speed_predictions(model.predict(ktsa_bim_X, verbose=0), test_context, scaler)
        inv_random_bim = invert_speed_predictions(model.predict(random_bim_X, verbose=0), test_context, scaler)

        # 计算当前的 RMSE
        rmse_ktsa_fgsm = np.sqrt(mean_squared_error(inv_y, inv_ktsa_fgsm))
        rmse_random_fgsm = np.sqrt(mean_squared_error(inv_y, inv_random_fgsm))
        rmse_ktsa_bim = np.sqrt(mean_squared_error(inv_y, inv_ktsa_bim))
        rmse_random_bim = np.sqrt(mean_squared_error(inv_y, inv_random_bim))

        # 记录增量 (increase over Clean)
        ktsa_fgsm_rmse_inc.append(rmse_ktsa_fgsm - clean_rmse)
        random_fgsm_rmse_inc.append(rmse_random_fgsm - clean_rmse)
        ktsa_bim_rmse_inc.append(rmse_ktsa_bim - clean_rmse)
        random_bim_rmse_inc.append(rmse_random_bim - clean_rmse)

        print(
            f"KTSA-FGSM inc: {rmse_ktsa_fgsm - clean_rmse:.3f} | Random-FGSM inc: {rmse_random_fgsm - clean_rmse:.3f}")
        print(
            f"KTSA-BIM inc : {rmse_ktsa_bim - clean_rmse:.3f} | Random-BIM inc : {rmse_random_bim - clean_rmse:.3f}\n")

    # 3. 绘制并保存 FGSM 对比图
    fig_fgsm = plt.figure(figsize=(10, 6))
    plt.plot(top_ratios, ktsa_fgsm_rmse_inc, marker='o', linewidth=2, color='red', label="KTSA-FGSM")
    plt.plot(top_ratios, random_fgsm_rmse_inc, marker='s', linewidth=2, linestyle='--', color='gray',
             label="Random-Step-FGSM")
    plt.title("KTSA-FGSM vs Random-Step-FGSM Performance", fontsize=14)
    plt.ylabel("RMSE increase over Clean", size=12)
    plt.xlabel("Top Ratio (Key Timesteps Proportion)", size=12)
    plt.xticks(top_ratios, [f"{int(r * 100)}%" for r in top_ratios])
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    fig_fgsm.savefig("../Images/KTSA_FGSM_vs_Random_FGSM_across_top_ratios.png", bbox_inches='tight')
    plt.close(fig_fgsm)

    # 4. 绘制并保存 BIM 对比图
    fig_bim = plt.figure(figsize=(10, 6))
    plt.plot(top_ratios, ktsa_bim_rmse_inc, marker='o', linewidth=2, color='blue', label="KTSA-BIM")
    plt.plot(top_ratios, random_bim_rmse_inc, marker='s', linewidth=2, linestyle='--', color='gray',
             label="Random-Step-BIM")
    plt.title("KTSA-BIM vs Random-Step-BIM Performance", fontsize=14)
    plt.ylabel("RMSE increase over Clean", size=12)
    plt.xlabel("Top Ratio (Key Timesteps Proportion)", size=12)
    plt.xticks(top_ratios, [f"{int(r * 100)}%" for r in top_ratios])
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    fig_bim.savefig("../Images/KTSA_BIM_vs_Random_BIM_across_top_ratios.png", bbox_inches='tight')
    plt.close(fig_bim)

    print("Experiment completed. Check the '../Images/' folder for the plotted graphs.")