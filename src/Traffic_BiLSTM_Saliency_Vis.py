import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import traffic_data, compute_time_step_saliency, rmse
from model import setup_bilstm_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'Trained models', 'Traffic_regression_BiLSTM.h5')

# 加载数据
train_X, train_y, val_X, val_y, test_X, test_y, scaler = traffic_data()

if os.path.isfile(model_path):
    model = setup_bilstm_model(train_X, 100)
    model.load_weights(model_path)

    sample_index = 0
    single_X = test_X[sample_index:sample_index + 1]
    single_Y = test_y[sample_index:sample_index + 1]

    grad_np, saliency = compute_time_step_saliency(model, rmse, single_X, single_Y)

    saliency_scores = saliency[0]
    time_steps = len(saliency_scores)

    top_ratio = 0.1
    top_k = max(1, int(np.ceil(time_steps * top_ratio)))

    key_indices = np.argsort(saliency_scores)[-top_k:]

    fig_saliency = plt.figure(figsize=(10, 5))
    x_axis = np.arange(1, time_steps + 1)

    bars = plt.bar(x_axis, saliency_scores, color='steelblue', alpha=0.7, edgecolor='black')

    for idx in key_indices:
        bars[idx].set_color('crimson')
        bars[idx].set_edgecolor('black')
        bars[idx].set_hatch('//')

        plt.text(x_axis[idx], saliency_scores[idx], '★ Key', ha='center', va='bottom', color='red', fontweight='bold')

    plt.title(f"BiLSTM Gradient Saliency Distribution (Top {int(top_ratio * 100)}% Key Timesteps Highlighted)", fontsize=14)
    plt.xlabel("Time Step Index", fontsize=12)
    plt.ylabel("Saliency Score $S_{grad}(t)$", fontsize=12)
    plt.xticks(x_axis)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    output_path = os.path.join(BASE_DIR, 'Images', 'BiLSTM_Sample_Gradient_Saliency_with_Key_Timesteps.png')
    fig_saliency.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig_saliency)

    print(f"Saliency visualization saved to: {output_path}")
else:
    print("Model file not found: %s" % model_path)
