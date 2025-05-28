import numpy as np
import matplotlib.pyplot as plt

epochs = np.arange(1, 21)  # 20 epochs

# 產生模擬數據 (長度都要20)
train_loss = np.exp(-epochs / 15) + 0.03 * np.random.randn(20)
val_loss = np.exp(-epochs / 13) + 0.035 * np.random.randn(20) + 0.05
val_psnr = 20 + 8 * (1 - np.exp(-epochs / 18)) + 0.4 * np.random.randn(20)

plt.figure(figsize=(12,4))

# Training Loss
plt.subplot(1,3,1)
plt.plot(epochs, train_loss, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.legend()

# Validation Loss
plt.subplot(1,3,2)
plt.plot(epochs, val_loss, color='orange', label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss Curve')
plt.grid(True)
plt.legend()

# Validation PSNR
plt.subplot(1,3,3)
plt.plot(epochs, val_psnr, color='green', label='Val PSNR')
plt.xlabel('Epoch')
plt.ylabel('PSNR (dB)')
plt.title('Validation PSNR Curve')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('fake_curves.png')
plt.show()
