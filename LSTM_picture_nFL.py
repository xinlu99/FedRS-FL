import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import ast

# 可配置的参数
tasknum = 10
NUM_ROUNDS = 10
NUM_CLIENTS = 5
BATCH_SIZE = 32
TIMESTEPS = 32  # 时间步长，需根据数据调整

# 生成输出文件名
output_filename = f'result_picture/picture_lstm{NUM_ROUNDS}_task{tasknum}_client{NUM_CLIENTS}.txt'
detailed_results_filename = f'result_picture/detailed_results_task{tasknum}_final_round.txt'

# 1. 读取CSV文件
input_filename = f'csv_picture/picture_output_task{tasknum}.csv'
df = pd.read_csv(input_filename)

# 打乱数据
df = df.sample(frac=1).reset_index(drop=True)

# 2. 数据预处理
scaler = MinMaxScaler()

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

# 将数据转换为PyTorch张量
train_data = torch.tensor(train_data, dtype=torch.float32)
test_data = torch.tensor(test_data, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.float32)

# 将数据和模型移至GPU（如有可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data = train_data.to(device)
test_data = test_data.to(device)
train_labels = train_labels.to(device)
test_labels = test_labels.to(device)

# LSTM模型定义
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.3)  # 添加Dropout
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out

# 初始化模型
input_size = train_data.shape[2]
hidden_size = 25  # 调整隐藏单元数
num_layers = 2  # 调整LSTM层数
output_size = 2

def train_model(model, train_data, train_labels, epochs=500):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    for epoch in range(epochs):
        outputs = model(train_data)
        loss = criterion(outputs, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

def evaluate_model(model, test_data, test_labels):
    model.eval()
    with torch.no_grad():
        predictions = model(test_data)
        loss = mean_squared_error(test_labels.cpu().numpy(), predictions.cpu().detach().numpy())
        mae = mean_absolute_error(test_labels.cpu().numpy(), predictions.cpu().detach().numpy())
    return loss, mae

results_list = []

# 进行多轮训练
for round_num in range(NUM_ROUNDS):
    print(f"Round {round_num + 1}/{NUM_ROUNDS}...")

    # 在每一轮中，所有客户端独立训练模型
    client_models = []
    client_losses = []

    for i in range(NUM_CLIENTS):
        print(f"Training client {i+1}/{NUM_CLIENTS}...")

        # 每个客户端初始化自己的模型
        client_model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

        # 模拟客户端数据 (在实际情况中，客户端会有自己的数据)
        client_train_data = train_data
        client_train_labels = train_labels

        # 训练客户端模型
        train_model(client_model, client_train_data, client_train_labels)

        # 保存客户端模型状态
        client_models.append(client_model.state_dict())

        # 测试客户端模型
        client_loss, client_mae = evaluate_model(client_model, test_data, test_labels)
        client_losses.append((client_loss, client_mae))

    # 聚合客户端模型（这里采用简单平均策略，你可以根据需要实现更复杂的聚合策略）
    global_model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    global_state_dict = global_model.state_dict()

    for key in global_state_dict.keys():
        global_state_dict[key] = torch.mean(torch.stack([client_model[key] for client_model in client_models]), dim=0)
    
    global_model.load_state_dict(global_state_dict)

    # 在每一轮结束时测试全局模型并记录结果
    global_loss, global_mae = evaluate_model(global_model, test_data, test_labels)

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
            predictions = global_model(test_data).cpu().detach().numpy()  # 使用 detach() 分离梯度
            for i, (pred, act) in enumerate(zip(predictions, test_labels.cpu().numpy())):
                f.write(f"{i+1}\t{pred[0]}\t{pred[1]}\t{act[0]}\t{act[1]}\n")
        print(f"Detailed predictions saved to {detailed_results_filename}")

# 保存最终训练结果
with open(output_filename, 'w') as f:
    f.write('\n'.join(results_list))

print(f"Training completed. Results saved to {output_filename}.")

