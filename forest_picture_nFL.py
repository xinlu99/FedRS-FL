import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import ast

# 可配置的参数
tasknum = 10
NUM_ROUNDS = 10  # 设置训练轮数
NUM_CLIENTS = 10
BATCH_SIZE = 32

# 生成输出文件名
output_filename = f'result_picture/picture_rf{NUM_ROUNDS}_task{tasknum}_client{NUM_CLIENTS}.txt'
detailed_results_filename = f'result_picture/detailed_results_task{tasknum}_final_round.txt'
prediction_results_filename = f'result_picture/prediction_results_task{tasknum}.txt'

# 1. 读取CSV文件
input_filename = f'csv_picture/picture_output_task{tasknum}.csv'
df = pd.read_csv(input_filename)

# 打乱数据
df = df.sample(frac=1).reset_index(drop=True)

# 2. 数据预处理
scaler = StandardScaler()

# 更新特征列名以匹配新数据
features = ['Concurrent task count', 'Total task size (MB)', 
            'New task start time (relative)', 'New task size (MB)', 
            'New task FLOPs (GFLOPs)', 'CPU frequency (GHz)']

# 对任务时间和操作数的列进行处理，将字符串转换为数值数组
df['Task start times'] = df['Task start times'].apply(ast.literal_eval)
df['Task operations (FLOPs)'] = df['Task operations (FLOPs)'].apply(ast.literal_eval)

# 对主要特征进行标准化处理
df[features] = scaler.fit_transform(df[features])

# 展开 'Task start times' 和 'Task operations' 列
task_start_times_expanded = np.array(df['Task start times'].tolist())
task_operations_expanded = np.array(df['Task operations (FLOPs)'].tolist())

# 合并主要特征和展开的列表特征
all_features = np.hstack((
    df[features].values,
    task_start_times_expanded,
    task_operations_expanded
))

# 提取标签
labels = df[['Total latency (excluding new task)', 'New task latency']].values

# 数据划分
train_data, test_data, train_labels, test_labels = train_test_split(
    all_features, labels, test_size=0.1, random_state=42
)

# 分配到各客户端
chunk_size = len(train_data) // NUM_CLIENTS

client_data = []
client_labels = []
for i in range(NUM_CLIENTS):
    start_idx = i * chunk_size
    end_idx = (i + 1) * chunk_size

    client_inputs = train_data[start_idx:end_idx]
    client_outputs = train_labels[start_idx:end_idx]

    client_data.append(client_inputs)
    client_labels.append(client_outputs)

# 使用随机森林回归器
def train_rf_model(X_train, y_train):
    rf_model = RandomForestRegressor(
        n_estimators=200,           # 增加树的数量
        max_depth=20,               # 设置最大深度
        min_samples_split=5,        # 增加最小样本拆分数
        min_samples_leaf=4,         # 增加最小样本叶节点数
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    return rf_model

# 初始化全局模型
global_rf_model = train_rf_model(train_data, train_labels)

results_list = []

# 进行多轮训练
for round_num in range(NUM_ROUNDS):
    print(f"Round {round_num + 1}/{NUM_ROUNDS}...")

    # 在每一轮中，所有客户端从全局模型开始训练
    client_models = []
    all_client_losses = []  # 用于存储所有客户端的损失

    for i in range(NUM_CLIENTS):
        print(f"Training client {i+1}/{NUM_CLIENTS}...")
        client_model = train_rf_model(client_data[i], client_labels[i])
        client_models.append(client_model)

        # 测试客户端模型
        client_preds = client_model.predict(test_data)
        loss = mean_squared_error(test_labels, client_preds)
        mae = mean_absolute_error(test_labels, client_preds)
        all_client_losses.append((loss, mae))

    # 绘制并保存客户端损失曲线
    plt.figure(figsize=(12, 6))
    for i, (loss, mae) in enumerate(all_client_losses):
        plt.plot(range(NUM_ROUNDS), [loss] * NUM_ROUNDS, label=f'Client {i+1} - Round {round_num+1}')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title(f'Client Losses During Training - Round {round_num+1}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'client_losses_round_{round_num+1}.png')

    # 聚合客户端模型（简单地取平均）
    global_rf_model = train_rf_model(train_data, train_labels)

    # 在每一轮结束时测试全局模型并记录结果
    global_preds = global_rf_model.predict(test_data)
    global_loss = mean_squared_error(test_labels, global_preds)
    global_mae = mean_absolute_error(test_labels, global_preds)

    print(f"\nGlobal Model Test Results after Round {round_num + 1}:")
    print(f"  Loss: {global_loss:.4f}")
    print(f"  MAE: {global_mae:.4f}")

    # 记录全局模型测试结果
    global_test_result_str = (f"\nGlobal Model Test Results after Round {round_num + 1}:\n"
                              f"  Loss: {global_loss:.4f}\n"
                              f"  MAE: {global_mae:.4f}")
    results_list.append(global_test_result_str)

    # 在最后一轮保存详细预测结果
    if round_num == NUM_ROUNDS - 1:
        with open(detailed_results_filename, 'w') as f:
            f.write("Index\tPredicted\tActual\n")
            for i, (pred, act) in enumerate(zip(global_preds, test_labels)):
                f.write(f"{i+1}\t{pred}\t{act}\n")
        print(f"Detailed predictions saved to {detailed_results_filename}")

# 保存最终训练结果
with open(output_filename, 'w') as f:
    f.write('\n'.join(results_list))

# 保存预测结果
with open(prediction_results_filename, 'w') as f:
    f.write("Index\tPredicted (Total Latency)\tActual (Total Latency)\tPredicted (New Task Latency)\tActual (New Task Latency)\n")
    for i, (pred, act) in enumerate(zip(global_preds, test_labels)):
        pred_total_latency, pred_new_task_latency = pred
        act_total_latency, act_new_task_latency = act
        f.write(f"{i+1}\t{pred_total_latency:.4f}\t{act_total_latency:.4f}\t{pred_new_task_latency:.4f}\t{act_new_task_latency:.4f}\n")

print(f"Training completed. Results saved to {output_filename}.")
print(f"Prediction results saved to {prediction_results_filename}.")

