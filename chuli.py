import csv

# 定义要提取的round值
num=9
target_rounds = [20, 40, 60, 80,100]

# 打开文件并读取内容
input_file = f'pic/picture_cnn2_FL100_task{num}_epoch20_bs64_learnrate0.001_client10.txt'
output_file = f'pic/rounds_row{num}.csv'

# 准备提取的数据
selected_data = []

with open(input_file, 'r') as file:
    lines = file.readlines()

    # 查找目标round的数据
    for line in lines[1:]:
        data = line.strip().split('\t')
        round_num = int(data[0])
        
        if round_num in target_rounds:
            selected_data.extend([data[1], data[4], data[7]])  # 依次追加 'loss of global', 'Max Loss of client', 'Avg Loss of client'

# 将提取的数据写入CSV文件
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # 只写一排数据
    writer.writerow(selected_data)

print(f"数据已保存到 {output_file}")

