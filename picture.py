import torch
import argparse
from PIL import Image
from fvcore.nn import FlopCountAnalysis

def load_model():
    # 从本地路径加载模型
    model = torch.hub.load('./yolov5', 'yolov5s', source='local', pretrained=True)
    return model

def compute_flops(model, input_size=(1, 3, 640, 640)):
    # 创建一个随机输入张量
    input_tensor = torch.randn(*input_size)
    # 计算模型的 FLOPs
    flops = FlopCountAnalysis(model, input_tensor)
    return flops.total()

def detect_objects(model, img):
    results = model(img)
    return results

def main(img_path):
    model = load_model()

    # 计算 FLOPs
    flops = compute_flops(model)
    print(f"Estimated FLOPs: {flops / 1e9:.2f} GFLOPs")

    # 执行对象检测
    img = Image.open(img_path)
    results = detect_objects(model, img)

    # 打印检测结果
    results.print()
    # 显示图像
    #results.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run YOLOv5 object detection.')
    parser.add_argument('img_path', type=str, help='Path to the input image')
    args = parser.parse_args()

    main(args.img_path)

