import os
from PIL import Image

folder_path = "/data2/jian/LangSplat/eval/masks_cv2"

for filename in os.listdir(folder_path):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        img_path = os.path.join(folder_path, filename)
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                print(f"{filename}: {width}x{height}")
        except Exception as e:
            print(f"Error reading {filename}: {e}")
