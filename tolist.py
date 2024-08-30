import csv
import re

# 输入txt文件路径
input_file = '/home/mxl/桌面/result_picture/picture_cnn2_FL100_task5_epoch20_bs64_learnrate0.001_client10.txt'
# 输出csv文件路径
output_file = 'output.csv'
# 初始化数据结构
# 初始化数据结构
# 初始化数据结构
data = []
global_results = {}
client_results = {}

with open(input_file, 'r') as infile:
    lines = infile.readlines()

    for line in lines:
        # 查找Global结果
        round_match = re.search(r'Global Model Test Results after Round (\d+):', line)
        if round_match:
            # 如果当前已经有global_results和client_results，保存它们
            if global_results and client_results:
                row = [
                    global_results['round'],
                    global_results.get('loss of global', ''),
                    global_results.get('MAE of global', ''),
                    global_results.get('MSE of global', ''),
                    client_results.get('Max Loss of client', ''),
                    client_results.get('Max MAE of client', ''),
                    client_results.get('Max MSE of client', ''),
                    client_results.get('Avg Loss of client', ''),
                    client_results.get('Avg MAE of client', ''),
                    client_results.get('Avg MSE of client', '')
                ]
                data.append(row)
                # 清空client_results以准备下一轮数据
                client_results = {}

            current_round = int(round_match.group(1))
            global_results = {'round': current_round}

        if 'Loss (MSE):' in line and 'Global' in global_results:
            global_results['loss of global'] = float(line.split(':')[1].strip())
        elif 'MAE:' in line and 'Global' in global_results:
            global_results['MAE of global'] = float(line.split(':')[1].strip())
        elif 'MSE:' in line and 'Global' in global_results:
            global_results['MSE of global'] = float(line.split(':')[1].strip())

        # 查找Client最大值和平均值
        client_match = re.search(r'Round (\d+) Client Test Results:', line)
        if client_match:
            current_round = int(client_match.group(1))
            if current_round == global_results.get('round'):
                client_results = {}

        if 'Max Loss (MSE):' in line:
            client_results['Max Loss of client'] = float(line.split(':')[1].strip())
        elif 'Max MAE:' in line:
            client_results['Max MAE of client'] = float(line.split(':')[1].strip())
        elif 'Max MSE:' in line:
            client_results['Max MSE of client'] = float(line.split(':')[1].strip())
        elif 'Avg Loss (MSE) across clients:' in line:
            client_results['Avg Loss of client'] = float(line.split(':')[1].strip())
        elif 'Avg MAE across clients:' in line:
            client_results['Avg MAE of client'] = float(line.split(':')[1].strip())
        elif 'Avg MSE across clients:' in line:
            client_results['Avg MSE of client'] = float(line.split(':')[1].strip())

    # 将最后一个轮次的数据添加到列表
    if global_results and client_results:
        row = [
            global_results['round'],
            global_results.get('loss of global', ''),
            global_results.get('MAE of global', ''),
            global_results.get('MSE of global', ''),
            client_results.get('Max Loss of client', ''),
            client_results.get('Max MAE of client', ''),
            client_results.get('Max MSE of client', ''),
            client_results.get('Avg Loss of client', ''),
            client_results.get('Avg MAE of client', ''),
            client_results.get('Avg MSE of client', '')
        ]
        data.append(row)

# 写入csv文件
header = [
    'round', 'loss of global', 'MAE of global', 'MSE of global',
    'Max Loss of client', 'Max MAE of client', 'Max MSE of client',
    'Avg Loss of client', 'Avg MAE of client', 'Avg MSE of client'
]

with open(output_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(header)
    for row in data:
        csvwriter.writerow(row)

print(f'数据已成功从 {input_file} 转换为 {output_file}')
          
