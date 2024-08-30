import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import ast

# 可配置的参数
tasknum = 10
NUM_ROUNDS = 10
NUM_CLIENTS = 10
BATCH_SIZE = 32
TIMESTEPS = 10  # 时间步长，需根据数据调整

# 生成输出文件名
output_filename = f'result_picture/picture_lstm{NUM_ROUNDS}_task{tasknum}_client{NUM_CLIENTS}.txt'
detailed_results_filename = f'result_picture/detailed_results_task{tasknum}_final_round.txt'

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

# 将数据转换为LSTM输入格式
def create_lstm_dataset(data, labels, timesteps):
    X, y = [], []
    for i in range(len(data) - timesteps + 1):
        X.append(data[i:i + timesteps])
        y.append(labels[i + timesteps - 1])
    return np.array(X), np.array(y)

# 提取标签
labels = df[['Total latency (excluding new task)', 'New task latency']].values

# 创建LSTM数据集
X, y = create_lstm_dataset(all_features, labels, TIMESTEPS)

# 数据划分
train_data, test_data, train_labels, test_labels = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# 使用LSTM模型
def train_lstm_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(2))  # 两个输出：Total latency 和 New task latency
    model.compile(optimizer='adam', loss='mse')
    return model

# 初始化全局模型
global_lstm_model = train_lstm_model(train_data, train_labels)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

results_list = []

# 进行多轮训练
for round_num in range(NUM_ROUNDS):
    print(f"Round {round_num + 1}/{NUM_ROUNDS}...")

    # 在每一轮中，所有客户端从全局模型开始训练
    client_models = []
    all_client_losses = []  # 用于存储所有客户端的损失

    for i in range(NUM_CLIENTS):
        print(f"Training client {i+1}/{NUM_CLIENTS}...")
        client_model = train_lstm_model(train_data, train_labels)
        client_model.fit(client_data[i], client_labels[i], epochs=10, batch_size=BATCH_SIZE, 
                         validation_split=0.1, callbacks=[early_stopping], verbose=0)
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
    global_lstm_model = train_lstm_model(train_data, train_labels)

    # 在每一轮结束时测试全局模型并记录结果
    global_preds = global_lstm_model.predict(test_data)
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
            f.write("Index\tPredicted Total Latency\tPredicted New Task Latency\tActual Total Latency\tActual New Task Latency\n")
            for i, (pred, act) in enumerate(zip(global_preds, test_labels)):
                f.write(f"{i+1}\t{pred[0]}\t{pred[1]}\t{act[0]}\t{act[1]}\n")
        print(f"Detailed predictions saved to {detailed_results_filename}")

# 保存最终训练结果
with open(output_filename, 'w') as f:
    f.write('\n'.join(results_list))

print(f"Training completed. Results saved to {output_filename}.")

