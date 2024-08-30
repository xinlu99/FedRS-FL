import csv

# 定义要提取的round值
target_rounds = [5,10,15,20]

# 准备输出的CSV文件
output_file = 'pic/mae——rounds_rows_all.csv'
#output_file = 'compute/mae—rounds_rows_com.csv'
# 打开输出文件准备写入
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # 遍历文件编号
    for num in range(1, 11):
        # 准备每个文件的输入文件名
        input_file = f'pic/picture_cnn2_FL100_task{num}_epoch20_bs64_learnrate0.001_client10.txt'
        #input_file = f'compute/fibon_cnn_FL100_task{num}_epoch20_bs64_learnrate0.001_client10.txt'
        # 准备提取的数据
        selected_data = []
        
        # 读取并处理文件
        with open(input_file, 'r') as file:
            lines = file.readlines()
        
            # 查找目标round的数据
            for line in lines[1:]:
                data = line.strip().split('\t')
                round_num = int(data[0])
                
                if round_num in target_rounds:
                    selected_data.extend([data[2], data[5], data[8]])  # 依次追加 'MAE of global', 'Max MAE of client', 'Avg MAE of client'
        
        # 将提取的数据写入CSV文件，作为一行
        writer.writerow(selected_data)

print(f"所有数据已保存到 {output_file}")

