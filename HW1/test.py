import os
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
import torchvision.transforms.functional as F
import re
import pickle

# 載入 class_to_idx（訓練時存的）
with open('dict_ResNet50.pkl', 'rb') as f:
    class_to_idx = pickle.load(f)

# 反轉：index → class label（資料夾名稱）
idx_to_class = {v: k for k, v in class_to_idx.items()}


print(idx_to_class)
# ---------- 自定義 SquarePad ----------
class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        return F.pad(image, (hp, vp, max_wh - w - hp, max_wh - h - vp), 0, 'constant')

# ---------- 自然排序用 ----------
def natural_key(string):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string)]

# ---------- 設定 ----------
num_classes = 100
input_size = 224
resize = 256
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = './model/model_ResNet50'
test_dir = './data/test'
output_csv = 'prediction.csv'

# ---------- 模型定義（保持和 train 一致） ----------
def initialize_model(n_classes):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, n_classes)
    )
    return model

# ---------- 載入模型 ----------
model = initialize_model(num_classes)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# ---------- 前處理 ----------
transform = transforms.Compose([
    SquarePad(),
    transforms.Resize(resize),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


image_names = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

pred_list = []
id_list = []

# ---------- 預測 ----------
for name in image_names:
    img_path = os.path.join(test_dir, name)
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        
        pred = torch.argmax(output, dim=1).item()
        pred_label = idx_to_class[pred]  # ← 得到實際的資料夾 label，例如 '17'

    id_list.append(os.path.splitext(name)[0])
    pred_list.append(pred_label)

# ---------- 輸出 CSV ----------
print("\n📚 idx_to_class label 對應表（模型 index → 資料夾名）:")
for idx in sorted(idx_to_class.keys()):
    print(f"Model Output Index: {idx:>3}  -->  Folder Label: {idx_to_class[idx]}")
df = pd.DataFrame({
    'image_name': id_list,
    'pred_label': pred_list
})
df.to_csv(output_csv, index=False)
print(f"✅ 已產生 {output_csv}，共 {len(df)} 筆預測")
