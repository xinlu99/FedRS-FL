import concurrent.futures
import time
import psutil
import os
import csv
import random
from datetime import datetime

def fibonacci(n):
    if n <= 1:
        return n, 0  # 基本的FLOPs为0
    else:
        a, flops_a = fibonacci(n-1)
        b, flops_b = fibonacci(n-2)
        return a + b, flops_a + flops_b + 1  # 每次加法操作计为1个FLOP

def run_computation(n, total_csv_file, task_id, count, start_time1):
    start_time = datetime.now()  # 记录任务的开始时间

    # 计算相对开始时间
    if start_time1:
        relative_start_time = (start_time - start_time1).total_seconds()
    else:
        relative_start_time = 0

    # 执行计算任务，计算第 n 个斐波那契数列，并记录FLOPs
    _, flops = fibonacci(n)

    end_time = datetime.now()  # 记录任务的结束时间
    task_latency = (end_time - start_time).total_seconds()  # 计算任务总延迟
    cpu_freq = psutil.cpu_freq().current / 1000  # 获取当前CPU频率（单位为GHz）

    with open(total_csv_file, 'a', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            task_id,
            count,
            n,  # 记录斐波那契数列的大小
            relative_start_time,
            task_latency,
            cpu_freq,  # 以GHz为单位记录CPU频率
            flops  # 记录FLOPs
        ])

def main(task_counts, it, iteration):
    # 确保目标文件夹存在
    os.makedirs(f"csv_fibon{task_counts}", exist_ok=True)
    total_csv_file = f"csv_fibon{task_counts}/fibo_{task_counts}_{it}_{iteration}.csv"
    delay_seconds = 0.01

    print(f"Running iteration {iteration} with {task_counts} concurrent tasks, {it} trials.")

    with open(total_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Task ID", "Task Number", "Fibonacci Size", "Relative Start Time (s)", "Task Latency (s)", "CPU Freq (GHz)", "FLOPs"])

    for trial in range(it):
        with concurrent.futures.ThreadPoolExecutor(max_workers=task_counts+1) as executor:
            futures = []
            start_time1 = None  # 初始化第一个任务的开始时间

            for idx in range(task_counts):
                time.sleep(random.uniform(0, 0.03))
                n = random.randint(25, 35)  # 斐波那契数列的大小随机取值，增加计算量
                task_id = f"{trial + 1}-{idx + 1}"  # 试验编号-任务编号
                
                if idx == 0:  # 第一个任务记录开始时间
                    start_time1 = datetime.now()

                futures.append(
                    executor.submit(run_computation, n, total_csv_file, task_id, task_counts, start_time1))
                print(f"Starting task: {task_id} with Fibonacci size {n}")

            time.sleep(random.uniform(0, 0.03))
            n = random.randint(25, 35)  # 斐波那契数列的大小随机取值，增加计算量
            task_id = f"{trial + 1}-{task_counts + 1}"  # 试验编号-新任务编号
            futures.append(
                executor.submit(run_computation, n, total_csv_file, task_id, task_counts, start_time1))
            print(f"Starting new task: {task_id} with Fibonacci size {n}")

            # 等待所有线程完成
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                    print(f"Completed task: {task_id}")
                except Exception as e:
                    print(f"Task execution error: {e}")

        print(f"Trial {trial + 1}/{it} completed for iteration {iteration}.")

if __name__ == "__main__":
    n = 20  # 设置生成 CSV 文件的次数
    it = 54  # 每次运行的试验次数
    task_counts = 6  # 并发任务（线程）数

    for iteration in range(1, n + 1):
        print(f"Running iteration {iteration}")
        main(task_counts, it, iteration)

