import numpy as np
import matplotlib.pyplot as plt

# 讀取資料
train_loss = np.load("train_losses.npy")
val_map50 = np.load("val_mAP50.npy")

# 建立 epoch 序列
epochs = np.arange(1, len(train_loss) + 1)

# 畫圖
plt.figure(figsize=(10, 5))

# 子圖 1：Train Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label="Train Loss", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.legend()

# 子圖 2：Validation mAP@0.5
plt.subplot(1, 2, 2)
plt.plot(epochs, val_map50, label="Val mAP@0.5", color="green")
plt.xlabel("Epoch")
plt.ylabel("mAP@0.5")
plt.title("Validation mAP@0.5")
plt.grid(True)
plt.legend()

plt.tight_layout()

# 儲存圖片
plt.savefig("training_curve.png", dpi=300)  # 你可以改成絕對路徑或存到 /mnt/data/
print("✅ 已儲存為 training_curve.png")
