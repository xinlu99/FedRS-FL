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
tasknum = 5
EPOCHS = 500
NUM_ROUNDS = 5  # 设置训练轮数
NUM_CLIENTS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
L2_REG = 0.001

# 生成输出文件名
output_filename = f'result_picture/picture_linear_FL{NUM_ROUNDS}_task{tasknum}_epoch{EPOCHS}_bs{BATCH_SIZE}_learnrate{LEARNING_RATE}_client{NUM_CLIENTS}.txt'
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

class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化全局模型
input_size = all_features.shape[1]
output_size = labels.shape[1]
global_model = LinearRegressionModel(input_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.RMSprop(global_model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG)

# 初始化保存内容的列表
results_list = []

# 进行多轮训练
for round_num in range(NUM_ROUNDS):
    print(f"Round {round_num + 1}/{NUM_ROUNDS}...")

    # 在每一轮中，所有客户端从全局模型开始训练
    client_models = []
    all_client_losses = []  # 用于存储所有客户端的损失

    for i, client_dataset in enumerate(client_data):
        print(f"Training client {i+1}/{NUM_CLIENTS}...")
        client_model = LinearRegressionModel(input_size, output_size).to(device)
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
                client_optimizer.zero_grad()
                outputs_pred = client_model(inputs)
                loss = criterion(outputs_pred, outputs)
                loss.backward()
                client_optimizer.step()
                epoch_loss += loss.item()
            
            # 每10个epoch打印一次Loss和MAE
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(train_loader)
                with torch.no_grad():
                    total_mae = 0.0
                    total_mse = 0.0
                    for inputs, outputs in train_loader:
                        inputs, outputs = inputs.to(device), outputs.to(device)
                        outputs_pred = client_model(inputs)
                        total_mae += torch.mean(torch.abs(outputs_pred - outputs)).item()
                        total_mse += torch.mean((outputs_pred - outputs) ** 2).item()
                    avg_mae = total_mae / len(train_loader)
                    avg_mse = total_mse / len(train_loader)
                print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.4f}, MAE: {avg_mae:.4f}, MSE: {avg_mse:.4f}")
            
            client_losses.append(epoch_loss / len(train_loader))
        
        client_models.append(client_model.state_dict())
        all_client_losses.append(client_losses)  # 记录损失值

    # 绘制并保存客户端损失曲线
    plt.figure(figsize=(12, 6))
    for i, losses in enumerate(all_client_losses):
        plt.plot(range(EPOCHS), losses, label=f'Client {i+1} - Round {round_num+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Client Losses During Training - Round {round_num+1}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'client_losses_round_{round_num+1}.png')

    # 初始化全局模型权重
    new_weights = dict.fromkeys(global_model.state_dict().keys())

    # 聚合客户端模型权重
    total_samples = sum(len(client_dataset) for client_dataset in client_data)

    # 遍历客户端权重，计算加权平均
    for param_name, param_tensor in global_model.state_dict().items():
        client_weight_sum = torch.zeros_like(param_tensor)
        for client_w in client_models:
            if param_name in client_w:
                client_weight_sum += client_w[param_name] * len(client_dataset) / total_samples
        new_weights[param_name] = client_weight_sum

    # 更新全局模型权重
    global_model.load_state_dict(new_weights)

    # 在每一轮结束时测试全局模型并记录结果
    global_model.eval()
    with torch.no_grad():
        test_loader = DataLoader(CustomDataset(test_data, test_labels), batch_size=BATCH_SIZE, shuffle=False)
        total_loss = 0.0
        total_mae = 0.0
        total_mse = 0.0
        num_batches = 0

        predictions = []
        actuals = []
        
        for inputs, outputs in test_loader:
            inputs, outputs = inputs.to(device), outputs.to(device)
            outputs_pred = global_model(inputs)
            loss = criterion(outputs_pred, outputs)
            total_loss += loss.item()
            total_mae += torch.mean(torch.abs(outputs_pred - outputs)).item()
            total_mse += torch.mean((outputs_pred - outputs) ** 2).item()
            num_batches += 1

            # 记录预测结果和实际值
            predictions.extend(outputs_pred.cpu().numpy())
            actuals.extend(outputs.cpu().numpy())

        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        avg_mse = total_mse / num_batches

        print(f"\nGlobal Model Test Results after Round {round_num + 1}:")
        print(f"  Loss (MSE): {avg_loss:.4f}")
        print(f"  MAE: {avg_mae:.4f}")
        print(f"  MSE: {avg_mse:.4f}")

        # 记录全局模型测试结果
        global_test_result_str = (f"\nGlobal Model Test Results after Round {round_num + 1}:\n"
                              f"  Loss (MSE): {avg_loss:.4f}\n"
                              f"  MAE: {avg_mae:.4f}\n"
                              f"  MSE: {avg_mse:.4f}")
        results_list.append(global_test_result_str)

        # 在最后一轮保存详细预测结果
        if round_num == NUM_ROUNDS - 1:
            with open(detailed_results_filename, 'w') as f:
                f.write("Index\tPredicted\tActual\n")
                for i, (pred, act) in enumerate(zip(predictions, actuals)):
                    f.write(f"{i+1}\t{pred}\t{act}\n")
            print(f"Detailed predictions saved to {detailed_results_filename}")

# 保存最终训练结果
with open(output_filename, 'w') as f:
    f.write('\n'.join(results_list))

print(f"Training completed. Results saved to {output_filename}.")

