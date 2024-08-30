import os
import pandas as pd
import csv

# 输入的CSV文件夹路径
num=6
input_folder = f'csv_fibon{num}/'

# 输出的CSV文件路径和文件名
output_csv = f'1csv_fibon/fibon_output_task{num}.csv'

# 初始化输出数据列表
output_data = []

# 遍历处理每个CSV文件
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        print(filename)
        input_csv = os.path.join(input_folder, filename)
        
        # 读取输入的CSV文件并处理数据
        df = pd.read_csv(input_csv)
        
        # 确保DataFrame按照 'Relative Start Time (s)' 排序
        df.sort_values(by='Relative Start Time (s)', inplace=True)
        
        # 遍历处理每个试验的数据
        for _, group in df.groupby(df['Task ID'].str.split('-', expand=True)[0]):
            # 获取第一个任务的开始时间
            first_task_start_time = group.iloc[0]['Relative Start Time (s)']
            
            # 获取新任务的信息
            new_task = group.iloc[-1]
            new_task_start_time = new_task['Relative Start Time (s)']
            new_task_latency = new_task['Task Latency (s)']
            
            # 检查新任务的开始时间是否在其他任务的范围内
    #        if new_task_start_time > (first_task_start_time + group.iloc[0]['Task Latency (s)']):
     #           print(f"Skipping entry where new task starts after first task ends: New task start time: #{new_task_start_time}, First task start time + latency: {first_task_start_time + group.iloc[0]['Task Latency #(s)']}")
           #     continue
            
            # 初始化总延迟时间和其他统计量
            total_latency = 0
            total_size = 0
            
            # 计算总延迟时间和总任务大小
            task_count = len(group) - 1  # 不包括新任务
            if task_count != num:
                print(f"task_count != {num}, task_count={task_count}")
                continue
            task_start_times = []
            task_operations = []
            
            for i in range(task_count):
                start_time = group.iloc[i]['Relative Start Time (s)']
                latency = group.iloc[i]['Task Latency (s)']
                total_latency += latency
                total_size += group.iloc[i]['Fibonacci Size']
                
                # 记录每个任务的开始时间和操作数
                task_start_times.append(round(start_time,4))
                task_operations.append(round(group.iloc[i]['FLOPs']/1e6,4))
            
            # 如果总延迟时间小于0，则跳过该条数据
            if total_latency < 0:
                print(f"Skipping entry with negative remaining time: {total_latency}")
                continue
            
            # 获取新任务的其他信息
            new_task_start = new_task_start_time - first_task_start_time
            new_task_size = new_task['Fibonacci Size']
            new_task_flops = round(new_task['FLOPs']/1e6,4)
            new_task_cpu_freq = new_task['CPU Freq (GHz)']
            
            # 准备输出数据的行，保留小数点后四位
            output_row = [
                round(task_count, 4),  # Concurrent task count (excluding the new task itself)
                round(total_size, 4),  # Total Fibonacci size of all tasks (excluding the new task)
                round(new_task_start, 4),  # Relative start time of new task
                round(total_latency, 4),  # Sum of remaining latency for original tasks
                round(new_task_latency, 4),  # Latency of new task
                round(new_task_size, 4),  # Fibonacci size of new task
                round(new_task_flops, 4),  # FLOPs of new task
                round(new_task_cpu_freq, 4),  # CPU frequency of new task
                task_start_times,  # Start times of all tasks
                task_operations  # Operations (FLOPs) of all tasks
            ]
            
            # 将处理后的数据行加入输出数据列表
            output_data.append(output_row)

# 写入输出的CSV文件
with open(output_csv, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow([
        "Concurrent task count",
        "Total Fibonacci size",
        "New task start time (relative)",
        "Total latency (excluding new task)",
        "New task latency",
        "New task Fibonacci size",
        "New task FLOPs",
        "CPU frequency (GHz)",
        "Task start times",  # 新添加的列，用于记录每个任务的开始时间
        "Task operations (FLOPs)"  # 新添加的列，用于记录每个任务的操作数
    ])
    csv_writer.writerows(output_data)

print(f"Processed data saved to {output_csv}")

