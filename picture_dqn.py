import concurrent.futures
import time
import psutil
import os
import csv
import random
import torch
from PIL import Image
import pandas as pd
from datetime import datetime  # 导入 datetime

# 定义一些辅助函数
def get_image_size(image_file):
    return round(os.path.getsize(image_file) / (1024 * 1024), 4)

def get_model_flops():
    return 8.22 * 10 ** 9  # 假设 YOLOv5s 的 FLOPs 是 8.22 GFLOPs (根据你的实际结果调整)

def run_detection(cfg_file, weight_file, folder_path, total_csv_file, task_id, task_number):
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                   os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.jpg')]

    total_images = len(image_files)
    total_data_size = 0
    total_flops = 0

    start_time1 = datetime.now()

    model = torch.hub.load('./yolov5', 'yolov5s', source='local', pretrained=True)

    for image_file in image_files:
        img = Image.open(image_file)
        print(f"PICTURE: {task_id} {image_file}")
        img = img.convert('RGB')
        results = model(img)  # 运行 YOLOv5 进行检测

        image_size = get_image_size(image_file)
        image_flops = get_model_flops()

        total_data_size += image_size
        total_flops += image_flops

    end_time1 = datetime.now()
    task_latency = round((end_time1 - start_time1).total_seconds(), 4)

    cpu_freq = round(psutil.cpu_freq().current / 1000, 4)

    with open(total_csv_file, 'a', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            task_id,
            task_number,
            task_latency,
            cpu_freq,
            round(total_data_size, 4),
            round(total_flops / 1e9, 4)  # 输出为 GFLOPs
        ])

def run_existing_tasks(task_count, selected_folders, total_csv_file):
    with concurrent.futures.ThreadPoolExecutor(max_workers=task_count) as executor:
        futures = []
        for idx in range(task_count):
            task_id = f"Existing-{idx + 1}"
            futures.append(
                executor.submit(run_detection, None, None, selected_folders[idx], total_csv_file, task_id, task_count))
            print(f"Starting existing task: {task_id} {selected_folders[idx]}")

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
                print(f"Completed existing task.")
            except Exception as e:
                print(f"Task execution error: {e}")

def run_with_new_task(task_count, selected_folders, total_csv_file, folder_paths):
    with concurrent.futures.ThreadPoolExecutor(max_workers=task_count + 1) as executor:
        futures = []
        for idx in range(task_count):
            task_id = f"WithNew-{idx + 1}"
            futures.append(
                executor.submit(run_detection, None, None, selected_folders[idx], total_csv_file, task_id, task_count))
            print(f"Starting task: {task_id} {selected_folders[idx]}")

        # 新任务使用一个新的随机文件夹
        selected_folder_new_task = random.choice(folder_paths)
        task_id = f"WithNew-{task_count + 1}"
        futures.append(
            executor.submit(run_detection, None, None, selected_folder_new_task, total_csv_file, task_id, task_count))
        print(f"Starting new task: {task_id} {selected_folder_new_task}")

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
                print(f"Completed task: {task_id}")
            except Exception as e:
                print(f"Task execution error: {e}")

def parse_csv_file(csv_file):
    data = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)  # Skip the header row
        for row in reader:
            data.append(row)
    return data

def process_data(task_count, run_times, n):
    base_dir = f"csv_picture_{task_count}"
    results = []

    for iterator in range(1, n):
        csv_file = f"{base_dir}/{task_count}_{run_times}_{iterator}.csv"
        if os.path.exists(csv_file):
            data = parse_csv_file(csv_file)

            for run_id in range(run_times):
                data_size = 0
                flops = 0
                new_task_latency = 0
                latency_with_new_task = 0
                latency_without_new_task = 0

                existing_tasks_data = data[run_id * task_count:(run_id + 1) * task_count]
                all_tasks_data = data[(run_id * task_count):(run_id + 1) * (task_count + 1)]

                # 计算不加新任务的总latency
                for task_data in existing_tasks_data:
                    latency_without_new_task += float(task_data[2])
                    data_size += float(task_data[4])
                    flops += float(task_data[5])

                # 计算加入新任务的总latency
                for task_data in all_tasks_data:
                    latency_with_new_task += float(task_data[2])

                # 新任务的latency
                new_task_latency = float(all_tasks_data[-1][2])

                results.append({
                    'Task Number': task_count,
                    'Data Size (MB)': round(data_size, 4),
                    'FLOPs (GFLOPs)': round(flops, 4),
                    'New Task Latency (s)': round(new_task_latency, 4),
                    'Total Latency with New Task (s)': round(latency_with_new_task, 4),
                    'Total Latency without New Task (s)': round(latency_without_new_task, 4)
                })

    return pd.DataFrame(results)

def main(task_count, iterator, run_times, n):
    folder_paths = [f"data/folder{i}" for i in range(1, 10)]  # 生成文件夹路径
    dictory_path = f"csv_picture_dqn_{task_count}"

    os.makedirs(dictory_path, exist_ok=True)
    total_csv_file = f"csv_picture_dqn_{task_count}/{task_count}_{run_times}_{iterator}.csv"

    print("Running tasks concurrently...")

    with open(total_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Task ID", "Task Number", "Task Latency (s)", "CPU Freq (GHz)",
             "Total Data Size (MB)", "Total FLOPs (GFLOPs)"])

    all_selected_folders = []

    for run_id in range(run_times):
        # 如果是第一次运行，随机选择文件夹并存储
        if len(all_selected_folders) < run_id * task_count + task_count:
            selected_folders = random.sample(folder_paths, task_count)
            all_selected_folders.extend(selected_folders)
        else:
            # 否则，复用之前存储的文件夹列表
            selected_folders = all_selected_folders[run_id * task_count:(run_id + 1) * task_count]

        print(f"\n--- Run {run_id + 1}: Running existing tasks without new task ---")
        run_existing_tasks(task_count, selected_folders, total_csv_file)

        print(f"\n--- Run {run_id + 1}: Running tasks with new task ---")
        run_with_new_task(task_count, selected_folders, total_csv_file, folder_paths)

        print(f"Detection completed for {run_id + 1} runs with {task_count} concurrent tasks.")

    df = process_data(task_count, run_times, n)
    output_file = f"consolidated_results_{task_count}_tasks.csv"
    df.to_csv(output_file, index=False)
    print(f"Data consolidation complete. Results saved to {output_file}.")

if __name__ == "__main__":
    task_count = 1  # 设置每次运行的并发任务数量
    run_times = 10  # 设置每个并发任务数量下的运行次数
    n = 51

    for iterator in range(1, n):
        print(f"Running with {task_count} concurrent tasks")
        main(task_count, iterator, run_times, n)
        
    task_count = 2  # 设置每次运行的并发任务数量
    run_times = 51  # 设置每个并发任务数量下的运行次数
    n = 10
    
    for iterator in range(1, n):
        print(f"Running with {task_count} concurrent tasks")
        main(task_count, iterator, run_times, n)

    task_count = 3  # 设置每次运行的并发任务数量
    run_times = 51  # 设置每个并发任务数量下的运行次数
    n = 10
    
    for iterator in range(1, n):
        print(f"Running with {task_count} concurrent tasks")
        main(task_count, iterator, run_times, n)

    task_count = 4  # 设置每次运行的并发任务数量
    run_times = 51  # 设置每个并发任务数量下的运行次数
    n = 10

    for iterator in range(1, n):
        print(f"Running with {task_count} concurrent tasks")
        main(task_count, iterator, run_times, n)
        
    task_count = 5  # 设置每次运行的并发任务数量
    run_times = 51  # 设置每个并发任务数量下的运行次数
    n = 10

    for iterator in range(1, n):
        print(f"Running with {task_count} concurrent tasks")
        main(task_count, iterator, run_times, n)

