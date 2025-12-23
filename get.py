import os
from modelscope import snapshot_download
import shutil

model_dir = snapshot_download('mistralai/Ministral-3-3B-Instruct-2512')
print(f"Model downloaded to: {model_dir}")

# 下载后的默认路径
default_model_dir = model_dir  # 获取下载路径

# 目标路径
target_model_dir = './mistralai'

# 移动文件夹
shutil.move(default_model_dir, target_model_dir)

print(f"Model moved to: {target_model_dir}")