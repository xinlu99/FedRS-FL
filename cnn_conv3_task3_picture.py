import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import ast

# 可配置的参数
tasknum=1
EPOCHS = 200
NUM_CLIENTS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
L2_REG = 0.001

# 生成输出文件名
output_filename = f'result_picture/picture_cnn_conv1_task{tasknum}_{EPOCHS}_{BATCH_SIZE}_{LEARNING_RATE}_{NUM_CLIENTS}.txt'

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
total_data_size = len(df)
train_ratio = 0.9
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
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.LeakyReLU(0.1)
        
        # 计算卷积层输出的尺寸
        self._to_linear = None
        self.convs = nn.Sequential(self.conv1, self.relu1, self.conv2, self.relu2)
        self._calculate_conv_output_size(input_size)
        
        self.fc1 = nn.Linear(self._to_linear, 64)
        self.relu3 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(64, 2)
        self.l2_reg = L2_REG

    def _calculate_conv_output_size(self, input_size):
        with torch.no_grad():
            x = torch.zeros(1, input_size, 1)  # [batch_size, input_size, sequence_length]
            x = self.convs(x)
            self._to_linear = x.numel()

    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, 1)  # 展平，保留 batch_size 维度
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
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

# 在每个客户端训练本地模型并记录损失值
client_models = []
all_client_losses = []  # 用于存储所有客户端的损失
all_client_maes = []    # 用于存储所有客户端的MAE

for i, client_dataset in enumerate(client_data):
    print(f"Training and testing client {i+1}/{NUM_CLIENTS}...")
    client_model = SimpleCNN(input_size).to(device)
    client_model.load_state_dict(global_model.state_dict())
    client_model.train()

    train_loader = DataLoader(client_dataset, batch_size=BATCH_SIZE, shuffle=True)
    client_optimizer = optim.RMSprop(client_model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG)

    client_losses = []
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for inputs, outputs in train_loader:
            # 将数据迁移到GPU
            inputs, outputs = inputs.to(device), outputs.to(device)
            # 确保 inputs 的维度是 [batch_size, features]，转换为 [batch_size, channels, sequence_length]
            inputs = inputs.unsqueeze(1).transpose(1, 2)  # 添加 channels 维度并转置为 [batch_size, channels, sequence_length]
            client_optimizer.zero_grad()
            outputs_pred = client_model(inputs)
            loss = criterion(outputs_pred, outputs)
            loss.backward()
            client_optimizer.step()
            epoch_loss += loss.item()
        client_losses.append(epoch_loss / len(train_loader))
    
    client_models.append(client_model.state_dict())
    all_client_losses.append(client_losses)  # 记录损失值

    # 在每个客户端上进行测试
    client_model.eval()
    with torch.no_grad():
        test_loader = DataLoader(CustomDataset(test_data, test_labels), batch_size=BATCH_SIZE, shuffle=False)
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        for inputs, outputs in test_loader:
            # 将数据迁移到GPU
            inputs, outputs = inputs.to(device), outputs.to(device)
            # 确保 inputs 的维度是 [batch_size, features]，转换为 [batch_size, channels, sequence_length]
            inputs = inputs.unsqueeze(1).transpose(1, 2)  # 添加 channels 维度并转置为 [batch_size, channels, sequence_length]
            outputs_pred = client_model(inputs)
            loss = criterion(outputs_pred, outputs)
            total_loss += loss.item()
            total_mae += torch.mean(torch.abs(outputs_pred - outputs)).item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    print(f"\nClient {i+1} Test Results:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  MAE: {avg_mae:.4f}")
    
    test_result_str = (f"Client {i+1} Test Results:\n"
                       f"  Loss: {avg_loss:.4f}\n"
                       f"  MAE: {avg_mae:.4f}")
    results_list.append(test_result_str)

# 绘制并保存客户端损失曲线
plt.figure(figsize=(12, 6))
for i, losses in enumerate(all_client_losses):
    plt.plot(range(EPOCHS), losses, label=f'Client {i+1}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Client Losses During Training')
plt.legend()
plt.grid(True)
plt.savefig('client_losses.png')

# 初始化全局模型权重
new_weights = dict.fromkeys(global_model.state_dict().keys())

# 聚合客户端模型权重
total_samples = sum(len(client_dataset) for client_dataset in client_data)

# 遍历客户端权重，计算加权平均
for param_name, param_tensor in global_model.state_dict().items():
    client_weight_sum = torch.zeros_like(param_tensor)
    num_clients = len(client_models)
    
    for client_w in client_models:
        if param_name in client_w:
            client_weight_sum += client_w[param_name] * len(client_dataset) / total_samples
    
    new_weights[param_name] = client_weight_sum

# 更新全局模型权重
global_model.load_state_dict(new_weights)

# 在全局模型上进行测试并记录每条测试数据的预测结果
global_model.eval()
with torch.no_grad():
    test_loader = DataLoader(CustomDataset(test_data, test_labels), batch_size=BATCH_SIZE, shuffle=False)
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0

    predictions = []
    actuals = []
    
    for inputs, outputs in test_loader:
        inputs, outputs = inputs.to(device), outputs.to(device)
        inputs = inputs.unsqueeze(1).transpose(1, 2)  # 转置为 [batch_size, channels, sequence_length]
        outputs_pred = global_model(inputs)
        loss = criterion(outputs_pred, outputs)
        total_loss += loss.item()
        total_mae += torch.mean(torch.abs(outputs_pred - outputs)).item()
        num_batches += 1

        # 记录预测结果和实际值
        predictions.extend(outputs_pred.cpu().numpy())
        actuals.extend(outputs.cpu().numpy())

avg_loss = total_loss / num_batches
avg_mae = total_mae / num_batches
print(f"\nGlobal Model Test Results:")
print(f"  Loss: {avg_loss:.4f}")
print(f"  MAE: {avg_mae:.4f}")

# 记录每条测试数据的预测结果
test_results_str = "\n".join([
    f"Test Data {i+1}: Predicted: {pred}, Actual: {act}"
    for i, (pred, act) in enumerate(zip(predictions, actuals))
])

# 记录全局模型测试结果
global_test_result_str = (f"\nGlobal Model Test Results:\n"
                          f"  Loss: {avg_loss:.4f}\n"
                          f"  MAE: {avg_mae:.4f}")
results_list.append(test_results_str)
results_list.append(global_test_result_str)

# 保存训练结果
with open(output_filename, 'w') as f:
    f.write('\n'.join(results_list))

print(f"Training completed. Results saved to {output_filename}.")




