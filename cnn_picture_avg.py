import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 可配置的参数
tasknum = 2
EPOCHS = 20
NUM_ROUNDS = 100  # 设置训练轮数
NUM_CLIENTS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
L2_REG = 0.001

# 生成输出文件名
output_filename = f'result_picture/avg_picture_FL{NUM_ROUNDS}_task{tasknum}_epoch{EPOCHS}_bs{BATCH_SIZE}_learnrate{LEARNING_RATE}_client{NUM_CLIENTS}.txt'
detailed_results_filename = f'result_picture/detailed_avg_results_task{tasknum}_final_round.txt'

# 1. 读取CSV文件
input_filename = f'csv_picture/picture_avg_task{tasknum}.csv'
df = pd.read_csv(input_filename)

# 打乱数据
df = df.sample(frac=1).reset_index(drop=True)

# 2. 数据预处理
scaler = StandardScaler()

# 更新特征列名以匹配新数据
features = ['Concurrent task count', 'Total task size (MB)', 
            'New task start time (relative)', 'New task size (MB)', 
            'New task FLOPs (GFLOPs)', 'CPU frequency (GHz)', 
            'Average Task Start Time', 'Average Task Operations (FLOPs)']

# 对主要特征进行标准化处理
df[features] = scaler.fit_transform(df[features])

# 提取特征和标签
all_features = df[features].values
labels = df[['Total latency (excluding new task)', 'New task latency']].values

# 数据划分
total_data_size = len(df)
train_ratio = 0.85
train_size = int(total_data_size * train_ratio)
test_size = total_data_size - train_size

train_data = all_features[:train_size]
train_labels = labels[:train_size]
test_data = all_features[train_size:]
test_labels = labels[train_size:]

# 分配到各客户端
chunk_size = train_size // NUM_CLIENTS

class CustomDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.outputs = torch.tensor(outputs, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

client_data = []
for i in range(NUM_CLIENTS):
    start_idx = i * chunk_size
    end_idx = (i + 1) * chunk_size

    client_inputs = train_data[start_idx:end_idx]
    client_outputs = train_labels[start_idx:end_idx]

    client_data.append(
        CustomDataset(client_inputs, client_outputs)
    )

class SimpleCNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=1, padding=0)
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=1, padding=0)
        self.relu2 = nn.LeakyReLU(0.1)

        self._to_linear = None
        self.convs = nn.Sequential(self.conv1, self.relu1, self.conv2, self.relu2)
        self._calculate_conv_output_size(input_size)
        
        self.fc1 = nn.Linear(self._to_linear, 64)
        self.relu3 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(64, 64)
        self.relu4 = nn.LeakyReLU(0.1)
        self.fc3 = nn.Linear(64, 2)

    def _calculate_conv_output_size(self, input_size):
        with torch.no_grad():
            x = torch.zeros(1, input_size, 1)  # 假设输入的序列长度为1
            x = self.convs(x)
            self._to_linear = x.numel()

    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, 1)  # 展平，保留 batch_size 维度
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        return x

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化全局模型
input_size = all_features.shape[1]
global_model = SimpleCNN(input_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.RMSprop(global_model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG)

# 初始化保存内容的列表
results_list = []
header = ["round", "loss of global", "MAE of global", "MSE of global", 
          "Max Loss of client", "Max MAE of client", "Max MSE of client", 
          "Avg Loss of client", "Avg MAE of client", "Avg MSE of client"]
results_list.append('\t'.join(header))

# 进行多轮训练
for round_num in range(NUM_ROUNDS):
    print(f"Round {round_num + 1}/{NUM_ROUNDS}...")

    # 在每一轮中，所有客户端从全局模型开始训练
    client_models = []
    all_client_losses = []  # 用于存储所有客户端的损失

    for i, client_dataset in enumerate(client_data):
        print(f"Training client {i+1}/{NUM_CLIENTS}...")
        client_model = SimpleCNN(input_size).to(device)
        client_model.load_state_dict(global_model.state_dict())
        client_model.train()

        train_loader = DataLoader(client_dataset, batch_size=BATCH_SIZE, shuffle=True)
        client_optimizer = optim.RMSprop(client_model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG)

        client_losses = []
        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            for inputs, outputs in train_loader:
                inputs, outputs = inputs.to(device), outputs.to(device)
                inputs = inputs.unsqueeze(1).transpose(1, 2)
                client_optimizer.zero_grad()
                outputs_pred = client_model(inputs)
                loss = criterion(outputs_pred, outputs)
                loss.backward()
                client_optimizer.step()
                epoch_loss += loss.item()
            
            client_losses.append(epoch_loss / len(train_loader))
        
        client_models.append(client_model.state_dict())
        all_client_losses.append(client_losses)  # 记录损失值

    # 模型聚合部分
    new_global_model_state = dict(global_model.state_dict())  # 初始化全局模型的权重
    for key in new_global_model_state.keys():
        new_global_model_state[key] = torch.stack([client_models[i][key] for i in range(NUM_CLIENTS)], 0).mean(0)
    
    global_model.load_state_dict(new_global_model_state)

    # 对所有客户端在测试集上进行评估
    max_loss, max_mae, max_mse = 0.0, 0.0, 0.0
    total_loss, total_mae, total_mse = 0.0, 0.0, 0.0

    for i, client_dataset in enumerate(client_data):
        print(f"Testing client {i+1}/{NUM_CLIENTS}...")
        client_model = SimpleCNN(input_size).to(device)
        client_model.load_state_dict(client_models[i])
        client_model.eval()

        test_loader = DataLoader(CustomDataset(test_data, test_labels), batch_size=BATCH_SIZE, shuffle=False)
        client_total_loss = 0.0
        client_total_mae = 0.0
        client_total_mse = 0.0
        num_batches = 0

        for inputs, outputs in test_loader:
            inputs, outputs = inputs.to(device), outputs.to(device)
            inputs = inputs.unsqueeze(1).transpose(1, 2)
            outputs_pred = client_model(inputs)
            loss = criterion(outputs_pred, outputs)
            client_total_loss += loss.item()
            client_total_mae += torch.mean(torch.abs(outputs_pred - outputs)).item()
            client_total_mse += torch.mean((outputs_pred - outputs) ** 2).item()
            num_batches += 1

        avg_client_loss = client_total_loss / num_batches
        avg_client_mae = client_total_mae / num_batches
        avg_client_mse = client_total_mse / num_batches

        # 记录最大值
        max_loss = max(max_loss, avg_client_loss)
        max_mae = max(max_mae, avg_client_mae)
        max_mse = max(max_mse, avg_client_mse)

        # 累加各客户端的平均值
        total_loss += avg_client_loss
        total_mae += avg_client_mae
        total_mse += avg_client_mse

    avg_loss_all_clients = total_loss / NUM_CLIENTS
    avg_mae_all_clients = total_mae / NUM_CLIENTS
    avg_mse_all_clients = total_mse / NUM_CLIENTS

    # 在每一轮结束时测试全局模型并记录结果
    global_model.eval()
    with torch.no_grad():
        test_loader = DataLoader(CustomDataset(test_data, test_labels), batch_size=BATCH_SIZE, shuffle=False)
        total_loss = 0.0
        total_mae = 0.0
        total_mse = 0.0
        num_batches = 0

        for inputs, outputs in test_loader:
            inputs, outputs = inputs.to(device), outputs.to(device)
            inputs = inputs.unsqueeze(1).transpose(1, 2)  # 转置为 [batch_size, channels, sequence_length]
            outputs_pred = global_model(inputs)
            loss = criterion(outputs_pred, outputs)
            total_loss += loss.item()
            total_mae += torch.mean(torch.abs(outputs_pred - outputs)).item()
            total_mse += torch.mean((outputs_pred - outputs) ** 2).item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        avg_mse = total_mse / num_batches

    # 记录结果到表格中
    results_list.append(f"{round_num + 1}\t{avg_loss:.4f}\t{avg_mae:.4f}\t{avg_mse:.4f}\t"
                        f"{max_loss:.4f}\t{max_mae:.4f}\t{max_mse:.4f}\t"
                        f"{avg_loss_all_clients:.4f}\t{avg_mae_all_clients:.4f}\t{avg_mse_all_clients:.4f}")

    print(f"\nGlobal Model Test Results after Round {round_num + 1}:")
    print(f"  Global Loss (MSE): {avg_loss:.4f}, MAE: {avg_mae:.4f}, MSE: {avg_mse:.4f}")
    print(f"  Clients Max  -> Loss: {max_loss:.4f}, MAE: {max_mae:.4f}, MSE: {max_mse:.4f}")
    print(f"  Clients Avg  -> Loss: {avg_loss_all_clients:.4f}, MAE: {avg_mae_all_clients:.4f}, MSE: {avg_mse_all_clients:.4f}")

# 保存最终训练结果
with open(output_filename, 'w') as f:
    f.write('\n'.join(results_list))

print(f"Training completed. Results saved to {output_filename}.")

