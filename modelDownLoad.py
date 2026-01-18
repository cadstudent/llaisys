import os
# 1. 设置国内镜像（关键：解决下载慢的问题）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 导入下载函数（兼容所有huggingface-hub版本）
from huggingface_hub import snapshot_download

# 3. 下载模型（指定模型名和保存路径）
try:
    snapshot_download(
        repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",  # 模型名称
        local_dir="../Models/DeepSeek-R1-Distill-Qwen-1.5B",   # 保存路径
        # 可选：只下载必要文件，跳过无用缓存
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.bin"],
        ignore_patterns=["*.git*", "*.md", "LICENSE"]
    )
    print("模型下载完成！保存路径：../Models/DeepSeek-R1-Distill-Qwen-1.5B")
except Exception as e:
    print(f"下载失败：{e}")
    print("如果报错'Connection timed out'，请检查网络或更换镜像（比如把HF_ENDPOINT换成https://hf-mirror.com）")