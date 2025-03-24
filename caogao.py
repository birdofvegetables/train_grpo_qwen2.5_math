from modelscope import snapshot_download
import os

# 获取用户主目录
home_dir = os.path.expanduser("~")
# 定义新的缓存目录
new_cache_dir = os.path.join(home_dir, "model_cache")

# 下载模型并获取本地路径
local_path = snapshot_download('unsloth/Qwen2.5-3B', cache_dir=new_cache_dir)

# 打印下载到本地的绝对路径地址
print(f"模型下载到本地的绝对路径是: {local_path}")


