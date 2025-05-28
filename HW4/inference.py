import os
import torch
import numpy as np
from net.model import PromptIR
from PIL import Image

test_dir = 'test/degraded'
model_path = 'train_ckpt/epoch=10-step=66000.ckpt'
device = torch.device('cuda:0')
model = PromptIR(inp_channels=3, out_channels=3, dim=48, decoder=True).to(device)

# --- NEW: load lightning ckpt ---
ckpt = torch.load(model_path, map_location=device)
state_dict = ckpt['state_dict']
# (Optional) Remove 'net.' prefix if exists
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace('net.', '') if k.startswith('net.') else k
    new_state_dict[new_key] = v
model.load_state_dict(new_state_dict)

model.eval()
results = {}

for fname in sorted(os.listdir(test_dir)):
    img = np.array(Image.open(os.path.join(test_dir, fname)).convert('RGB')).astype(np.float32) / 255.0
    x = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        y = model(x)
    out_img = y.squeeze(0).cpu().numpy().transpose(1,2,0)
    out_img = np.clip(out_img * 255, 0, 255).astype(np.uint8)
    # 這一行必須 .transpose(2,0,1) 讓 shape 是 (3, H, W)
    results[fname] = out_img.transpose(2, 0, 1)

np.savez('pred.npz', **results)
print(f'已存 pred.npz ({len(results)} images)')
