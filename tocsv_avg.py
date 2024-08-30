import os
import pandas as pd
import csv

# 输入的CSV文件夹路径
num = 9
input_folder = f'csv_picture_{num}/'

# 输出的CSV文件路径和文件名
output_csv = f'csv_picture/picture_avg_task{num}.csv'

# 初始化输出数据列表
output_data = []

# 遍历处理每个CSV文件
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        print(filename)
        input_csv = os.path.join(input_folder, filename)
        
        # 读取输入的CSV文件并处理数据
        df = pd.read_csv(input_csv)
        
        # 将 'Start Time (s)' 和 'End Time (s)' 转换为浮点数
        df['Start Time (s)'] = df['Start Time (s)'].astype(float)
        df['End Time (s)'] = df['End Time (s)'].astype(float)
        
        # 确保DataFrame按照 'Start Time (s)' 排序
        df.sort_values(by='Start Time (s)', inplace=True)
        
        # 遍历处理每个试验的数据
        for _, group in df.groupby(df['Task ID'].str.split('-', expand=True)[0]):
            # 获取第一个任务的开始和结束时间
            first_task_start_time = group.iloc[0]['Start Time (s)']
            first_task_end_time = group.iloc[0]['End Time (s)']
            
            # 获取新任务的信息
            new_task = group.iloc[-1]
            new_task_start_time = new_task['Start Time (s)']
            
            # 检查新任务的开始时间是否小于第一个任务的结束时间
            if new_task_start_time > first_task_end_time:
                print(f"Skipping entry where new task starts after first task ends: New task start time: {new_task_start_time}, First task end time: {first_task_end_time}")
                continue
            
            # 初始化总延迟时间和其他统计量
            total_latency = 0
            total_size = 0
            
            # 计算总延迟时间和总任务大小
            task_count = len(group) - 1  # 不包括新任务
            if task_count != num:
                print(f"task_count != task_count={task_count}")
                continue
            task_start_times = []
            task_operations = []
            
            for i in range(task_count):
                end_time = group.iloc[i]['End Time (s)']
                start_time = group.iloc[i]['Start Time (s)']
                latency = end_time - new_task_start_time
                total_latency += latency
                total_size += group.iloc[i]['Total Data Size (MB)']
                
                # 记录每个任务的开始时间和操作数
                task_start_times.append(start_time)
                task_operations.append(group.iloc[i]['Total FLOPs (GFLOPs)'])
            
            # 如果总延迟时间小于0，则跳过该条数据
            if total_latency < 0:
                print(f"Skipping entry with negative remaining time: {total_latency}")
                continue
            
            # 获取新任务的其他信息
            new_task_start = new_task_start_time - first_task_start_time
            new_task_size = new_task['Total Data Size (MB)']
            new_task_latency = new_task['Task Latency (s)']
            new_task_flops = new_task['Total FLOPs (GFLOPs)']
            new_task_cpu_freq = new_task['CPU Freq (GHz)']
            
            # 计算平均开始时间和平均操作数
            avg_start_time = sum(task_start_times) / len(task_start_times)
            avg_operations = sum(task_operations) / len(task_operations)
            
            # 准备输出数据的行，保留小数点后四位
            output_row = [
                round(task_count, 4),  # Concurrent task count (excluding the new task itself)
                round(total_size, 4),  # Total size of all tasks (excluding the new task)
                round(new_task_start, 4),  # Relative start time of new task
                round(total_latency, 4),  # Sum of remaining latency for original tasks
                round(new_task_latency, 4),  # Latency of new task
                round(new_task_size, 4),  # Size of new task
                round(new_task_flops, 4),  # FLOPs of new task
                round(new_task_cpu_freq, 4),  # CPU frequency of new task
                round(avg_start_time, 4),  # 平均开始时间
                round(avg_operations, 4)  # 平均操作数
            ]
            
            # 将处理后的数据行加入输出数据列表
            output_data.append(output_row)

# 写入输出的CSV文件
with open(output_csv, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow([
        "Concurrent task count",
        "Total task size (MB)",
        "New task start time (relative)",
        "Total latency (excluding new task)",
        "New task latency",
        "New task size (MB)",
        "New task FLOPs (GFLOPs)",
        "CPU frequency (GHz)",
        "Average Task Start Time",  # 修改后的列，用于记录平均任务开始时间
        "Average Task Operations (FLOPs)"  # 修改后的列，用于记录平均任务操作数
    ])
    csv_writer.writerows(output_data)

print(f"Processed data saved to {output_csv}")

