import torch

# 加载checkpoint查看内容
ckpt_path = "/home/shiyao/EEG/LaBraM/checkpoints/vqnsp.pth"  # 替换为实际的checkpoint路径
checkpoint = torch.load(ckpt_path, map_location='cpu')

# 打印checkpoint的键
print("Checkpoint keys:", checkpoint.keys())

# 如果有model键，打印模型结构
if 'model' in checkpoint:
    print("\nModel state_dict keys:")
    for key in checkpoint['model'].keys():
        print(key)