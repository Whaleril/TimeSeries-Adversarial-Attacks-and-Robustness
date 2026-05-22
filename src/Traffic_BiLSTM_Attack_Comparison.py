import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 导入所需的攻击方法和工具函数
from utils import traffic_data, fgsm, bim, ktsa_fgsm, ktsa_bim, rmse
from model import setup_bilstm_model

# 设置随机种子，保证每次运行结果一致
np.random.seed(42)

# 模型路径
bilstm_model_path = r'D:\IDEA\BiLSTM\TimeSeries-Adversarial-Attacks-and-Robustness\Trained models\Traffic_regression_BiLSTM.h5'

# 攻击参数
epsilon = 0.2
alpha = 0.001
iterations = 200
top_ratio = 0.2  # 关键时间步比例

# 加载数据
train_X, train_y, val_X, val_y, test_X, test_y, scaler = traffic_data()

def invert_speed_predictions(predictions, test_context, scaler):
    inv_pred = np.concatenate((test_context, predictions), axis=1)
    inv_pred = scaler.inverse_transform(inv_pred)
    return inv_pred[:, 2]

# 加载BiLSTM模型
bilstm_model = setup_bilstm_model(train_X, 100)
bilstm_model.load_weights(bilstm_model_path)

# 计算Clean状态下的预测结果
test_context = test_X[:, -1, :2]
bilstm_yhat = bilstm_model.predict(test_X, verbose=0)
bilstm_test_y_reshaped = test_y.reshape((len(test_y), 1))
bilstm_inv_yhat = invert_speed_predictions(bilstm_yhat, test_context, scaler)
bilstm_inv_y = invert_speed_predictions(bilstm_test_y_reshaped, test_context, scaler)

# 执行各种攻击
print("Executing attacks...")

# FGSM攻击
fgsm_adv_X, _ = fgsm(X=test_X, Y=test_y, model=bilstm_model, loss_fn=rmse, epsilon=epsilon)
fgsm_pred = bilstm_model.predict(fgsm_adv_X, verbose=0)
fgsm_inv_pred = invert_speed_predictions(fgsm_pred, test_context, scaler)

# BIM攻击
bim_adv_X, _ = bim(X=test_X, Y=test_y, model=bilstm_model, loss_fn=rmse, epsilon=epsilon, alpha=alpha, I=iterations)
bim_pred = bilstm_model.predict(bim_adv_X, verbose=0)
bim_inv_pred = invert_speed_predictions(bim_pred, test_context, scaler)

# KTSA-FGSM攻击
ktsa_fgsm_adv_X, _, _, _ = ktsa_fgsm(X=test_X, Y=test_y, model=bilstm_model, loss_fn=rmse, epsilon=epsilon, top_ratio=top_ratio)
ktsa_fgsm_pred = bilstm_model.predict(ktsa_fgsm_adv_X, verbose=0)
ktsa_fgsm_inv_pred = invert_speed_predictions(ktsa_fgsm_pred, test_context, scaler)

# KTSA-BIM攻击
ktsa_bim_adv_X, _, _, _ = ktsa_bim(X=test_X, Y=test_y, model=bilstm_model, loss_fn=rmse, epsilon=epsilon, alpha=alpha, I=iterations, top_ratio=top_ratio)
ktsa_bim_pred = bilstm_model.predict(ktsa_bim_adv_X, verbose=0)
ktsa_bim_inv_pred = invert_speed_predictions(ktsa_bim_pred, test_context, scaler)

print("Generating comparison plot...")

# 生成对比图
fig = plt.figure(figsize=(15, 8))

# 只显示前200个时间步，保持与示例图一致
aa = [x for x in range(min(200, len(bilstm_inv_y)))]

# 绘制实际值
plt.plot(aa, bilstm_inv_y[:len(aa)], marker=".", label="actual", color="blue")

# 绘制Clean预测
plt.plot(aa, bilstm_inv_yhat[:len(aa)], label="prediction", color="red")

# 绘制FGSM预测
plt.plot(aa, fgsm_inv_pred[:len(aa)], label="fgsm prediction", color="green")

# 绘制BIM预测
plt.plot(aa, bim_inv_pred[:len(aa)], label="bim prediction", color="purple")

# 绘制KTSA-FGSM预测
plt.plot(aa, ktsa_fgsm_inv_pred[:len(aa)], label="ktsa-fgsm prediction", color="cyan")

# 绘制KTSA-BIM预测
plt.plot(aa, ktsa_bim_inv_pred[:len(aa)], label="ktsa-bim prediction", color="pink")

plt.ylabel("Speed", size=15)
plt.xlabel("Time step", size=15)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# 保存图表
output_path = r'D:\IDEA\BiLSTM\TimeSeries-Adversarial-Attacks-and-Robustness\Images\BiLSTM_Attack_Comparison.png'
plt.savefig(output_path, bbox_inches='tight')
plt.close(fig)

print(f"Comparison plot saved to: {output_path}")
print("Done!")
