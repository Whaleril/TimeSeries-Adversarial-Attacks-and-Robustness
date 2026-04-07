import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import traffic_data, compute_time_step_saliency, rmse
from model import setup_lstm_model

model_path = "../Trained models/Traffic_regression_LSTM.h5"

# 加载数据
train_X, train_y, val_X, val_y, test_X, test_y, scaler = traffic_data()

if os.path.isfile(model_path):
    # 加载模型
    model = setup_lstm_model(train_X, 100)
    model.load_weights(model_path)

    # 我们挑选测试集里的第一个样本来进行可视化展示
    sample_index = 0
    single_X = test_X[sample_index:sample_index + 1]
    single_Y = test_y[sample_index:sample_index + 1]

    # 计算梯度显著性
    grad_np, saliency = compute_time_step_saliency(model, rmse, single_X, single_Y)

    # 提取这 12 个时间步的显著性分数 (shape: [1, 12] -> [12])
    saliency_scores = saliency[0]
    time_steps = len(saliency_scores)

    # 设定 top ratio = 10%，算出要挑选几个关键时间步 (12 * 0.1 向上取整 = 2 个)
    top_ratio = 0.1
    top_k = max(1, int(np.ceil(time_steps * top_ratio)))

    # 找到分数最高的 top_k 个时间步的索引
    key_indices = np.argsort(saliency_scores)[-top_k:]

    # 开始绘图 (使用柱状图，效果最直观)
    fig_saliency = plt.figure(figsize=(10, 5))
    x_axis = np.arange(1, time_steps + 1)

    # 先画出所有时间步的柱子（默认深蓝色）
    bars = plt.bar(x_axis, saliency_scores, color='steelblue', alpha=0.7, edgecolor='black')

    # 把选中的关键时间步的柱子标红，并加上醒目的图案
    for idx in key_indices:
        bars[idx].set_color('crimson')
        bars[idx].set_edgecolor('black')
        bars[idx].set_hatch('//')  # 加点斜线阴影让它更明显

        # 在红柱子上方加一个小星星标记
        plt.text(x_axis[idx], saliency_scores[idx], '★ Key', ha='center', va='bottom', color='red', fontweight='bold')

    plt.title(f"Gradient Saliency Distribution (Top {int(top_ratio * 100)}% Key Timesteps Highlighted)", fontsize=14)
    plt.xlabel("Time Step Index", fontsize=12)
    plt.ylabel("Saliency Score $S_{grad}(t)$", fontsize=12)
    plt.xticks(x_axis)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # 保存图片
    output_path = "../Images/Sample_Gradient_Saliency_with_Key_Timesteps.png"
    fig_saliency.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig_saliency)

    print(f"Saliency visualization saved to: {output_path}")