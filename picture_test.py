import concurrent.futures
import subprocess
import time
import psutil
import os
from datetime import datetime
import csv
import random
import torch
from PIL import Image


def get_image_size(image_file):
    return round(os.path.getsize(image_file) / (1024 * 1024), 4)


def get_model_flops():
    # 假设 YOLOv5s 的 FLOPs 是 8.22 GFLOPs (根据你的实际结果调整)
    return 8.22 * 10 ** 9


def run_detection(cfg_file, weight_file, folder_path, total_csv_file, task_id, count, initial_start_time):
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                   os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.jpg')]

    total_images = len(image_files)
    total_data_size = 0
    total_flops = 0

    start_time1 = datetime.now()

    model = torch.hub.load('./yolov5', 'yolov5s', source='local', pretrained=True)

    for image_file in image_files:
        img = Image.open(image_file)
        # 图像预处理
        print(f"PICTURE: {task_id} {image_file}")
        img = img.convert('RGB')
        results = model(img)  # 运行 YOLOv5 进行检测

        image_size = get_image_size(image_file)
        image_flops = get_model_flops()

        total_data_size += image_size
        total_flops += image_flops

    end_time1 = datetime.now()
    task_latency = round((end_time1 - start_time1).total_seconds(), 4)

    relative_start_time = round((start_time1 - initial_start_time).total_seconds(), 4)
    relative_end_time = round((end_time1 - initial_start_time).total_seconds(), 4)
    cpu_freq = round(psutil.cpu_freq().current / 1000, 4)

    with open(total_csv_file, 'a', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            task_id,
            count,
            relative_start_time,
            relative_end_time,
            task_latency,
            cpu_freq,
            round(total_data_size, 4),
            round(total_flops / 1e9, 4)  # 输出为 GFLOPs
        ])


def main(task_count, iterator, run_times):
    folder_paths = [f"data/folder{i}" for i in range(1, 10)]  # 生成文件夹路径
    dictory_path = f"csv_picture_dqn_{task_count}"

    os.makedirs(dictory_path, exist_ok=True)
    total_csv_file = f"csv_picture_dqn_{task_count}/{task_count}_{run_times}_{iterator}.csv"

    print("Running tasks concurrently...")
    total_start_time = time.time()

    with open(total_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Task ID", "Task Number", "Start Time (s)", "End Time (s)", "Task Latency (s)", "CPU Freq (GHz)",
             "Total Data Size (MB)", "Total FLOPs (GFLOPs)", "Concurrent Task Count"])

    for run_id in range(run_times):
        initial_start_time = datetime.now()
        with concurrent.futures.ThreadPoolExecutor(max_workers=task_count + 1) as executor:
            futures = []
            for idx in range(task_count):
                time.sleep(random.uniform(0.01, 0.2))
                selected_folder = random.choice(folder_paths)  # 随机选择一个文件夹
                task_id = f"{run_id + 1}-{idx + 1}"
                futures.append(
                    executor.submit(run_detection, None, None, selected_folder, total_csv_file, task_id, task_count,
                                    initial_start_time))
                print(f"Starting task: {task_id} {selected_folder}")

            time.sleep(random.uniform(0.1, 0.3))
            selected_folder = random.choice(folder_paths)  # 随机选择一个文件夹
            task_id = f"{run_id + 1}-{idx + 2}"
            futures.append(
                executor.submit(run_detection, None, None, selected_folder, total_csv_file, task_id, task_count,
                                initial_start_time))
            print(f"Starting new task: {task_id} {selected_folder}")
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                    print(f"Completed task: {task_id}")
                except Exception as e:
                    print(f"Task execution error: {e}")

        total_end_time = time.time()
        total_elapsed_time = total_end_time - total_start_time
        print(f"Detection completed for {run_id + 1} runs with {task_count} concurrent tasks.")


if __name__ == "__main__":



        
    task_count = 9  # 设置每次运行的并发任务数量
    run_times = 35  # 设置每个并发任务数量下的运行次数
    n = 100

    for iterator in range(1, n):
        print(f"Running with {task_count} concurrent tasks")
        main(task_count, iterator, run_times)

