import json
import os

# JSON 文件路径
json_path = '/data2/jian/LangSplat/arguments/3dovs_feature_map_order.json'

# 读取 JSON 文件
with open(json_path, 'r') as f:
    config = json.load(f)

# 提取 file_order 列表
file_order = config.get('bed', [])

# 打印 file_order 内容
for filename in file_order:
    print(filename)
