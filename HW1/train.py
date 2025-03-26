# -*- coding: utf-8 -*-
import copy, os, pickle, time
import numpy as np
import torch
import torchvision.transforms.functional as F
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler  # ✅ 新增 AMP 模組
import csv
from collections import Counter  # ✅ For class weight calculation
import re


# ---------- 設定 ----------
dir_path = './data'
model = models.resnet50(pretrained=True)
model_path = './model/model_ResNet50'
dictionary_name = 'dict_ResNet50'

log_csv_path = './model/training_log.csv'
num_classes = 100
batch_size = 16
# 下次可以改小一點 16 或是 8
num_epochs = 100
input_size = 224
resize = 256
feature_extract = False
patience = 10  # early stopping 10 次過後沒有更好就跳出

save_plot_every = 10  # 每 N 個 epoch 存圖一次

def natural_key(string):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string)]


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------- 工具函式 ----------
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n\U0001F4CF Total trainable parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    if total_params > 100_000_000:
        print("❌ Parameter count exceeds 100M limit!")
    else:
        print("✅ Parameter count is within 100M limit.\n")


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# ---------- 加權 Loss 計算 ----------
def compute_class_weights(dataset):
    label_counts = Counter(dataset.targets)  # ✅ 使用內建 target list
    total = sum(label_counts.values())
    weights = [total / label_counts[i] if label_counts[i] > 0 else 0.0 for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float).to(device)


# ---------- Resume 機制 ----------
def try_resume(model, optimizer, scheduler):
    start_epoch = 0
    best_acc = 0.0
    if os.path.exists(model_path):
        print("🔄 Found checkpoint. Resuming training...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f"✅ Resumed from epoch {start_epoch}, best val acc = {best_acc:.4f}\n")
    else:
        print("ℹ️ No checkpoint found. Training from scratch.\n")
    return model, optimizer, scheduler, start_epoch, best_acc


def initialize_model(n_classes, feat_extract, use_pretrained=True):
    # model = models.resnet34(pretrained=use_pretrained)
    model = models.resnet50(pretrained=use_pretrained)
    set_parameter_requires_grad(model, feat_extract)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, n_classes)
    )
    return model


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        return F.pad(image, (hp, vp, max_wh - w - hp, max_wh - h - vp), 0, 'constant')


def train_model(model, dataloaders, criterion, optimizer, scheduler, n_epochs=25, start_epoch=0, best_acc=0.0):
    since = time.time()
    scaler = GradScaler()

    train_loss_history = []
    val_loss_history = []
    val_acc_history = []

    with open(log_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Val Accuracy'])

    best_model_wts = copy.deepcopy(model.state_dict())
    early_stop_counter = 0

    for epoch in range(start_epoch, n_epochs):
        print(f'Epoch {epoch + 1}/{n_epochs}\n{"-" * 10}')
        epoch_train_loss = None
        epoch_val_loss = None
        epoch_val_acc = None

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with autocast():
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                epoch_train_loss = epoch_loss
            else:
                val_loss_history.append(epoch_loss.cpu().item() if torch.is_tensor(epoch_loss) else epoch_loss)
                val_acc_history.append(epoch_acc.cpu().item() if torch.is_tensor(epoch_acc) else epoch_acc)
                epoch_val_loss = val_loss_history[-1]
                epoch_val_acc = val_acc_history[-1]

                # ✅ 每個 class 的準確率統計
                class_correct = [0] * num_classes
                class_total = [0] * num_classes

                for inputs, labels in dataloaders['val']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    with torch.no_grad():
                        with autocast():
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                    for label, pred in zip(labels, preds):
                        class_total[label.item()] += 1
                        if label.item() == pred.item():
                            class_correct[label.item()] += 1

                class_acc = [correct / total if total > 0 else 0.0
                            for correct, total in zip(class_correct, class_total)]

                # ✅ 每個 epoch 只印最差 15 個 class
                sorted_classes = sorted(enumerate(class_acc), key=lambda x: x[1])[:15]
                print("\n📉 Worst 15 Classes This Epoch:")
                for idx, acc in sorted_classes:
                    print(f"Class {idx:>3}: Acc = {acc:.4f} ({class_correct[idx]}/{class_total[idx]})")

                # ✅ 只有在 best acc 發生時，才存 classwise acc 到 CSV
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    early_stop_counter = 0

                    # 儲存當下 classwise 準確率
                    with open('./model/classwise_val_acc.csv', 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['class', 'correct', 'total', 'accuracy'])
                        for i in range(num_classes):
                            writer.writerow([i, class_correct[i], class_total[i], class_acc[i]])

                    # 儲存模型
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_acc': best_acc
                    }, model_path)
                else:
                    early_stop_counter += 1


        with open(log_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, epoch_train_loss, epoch_val_loss, epoch_val_acc])

        if (epoch + 1) % save_plot_every == 0:
            plot_training_curve(train_loss_history, val_loss_history, val_acc_history, epoch + 1)

        if early_stop_counter >= patience:
            print(f"\n❌ Early stopping triggered at epoch {epoch + 1}")
            break

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    plot_training_curve(train_loss_history, val_loss_history, val_acc_history, epoch + 1, final=True)
    model.load_state_dict(best_model_wts)
    return model, val_acc_history



# ---------- 畫圖 ----------
def plot_training_curve(train_loss, val_loss, val_acc, epoch, final=False):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Train Loss')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label='Val Loss')
    plt.plot(range(1, len(val_acc) + 1), val_acc, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    suffix = '_final' if final else f'_epoch{epoch}'
    path = f'./model/training_curve{suffix}.png'
    plt.savefig(path)
    plt.close()
    print(f"📈 Training curve saved to {path}")


# ---------- 主程式 ----------
if __name__ == '__main__':
    os.makedirs('./model', exist_ok=True)

    model_ft = initialize_model(num_classes, feature_extract, use_pretrained=True)
    model_ft = model_ft.to(device)

    count_parameters(model_ft)

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer_ft, T_0=10, T_mult=2)

    data_transforms = {
        'train': transforms.Compose([
        SquarePad(),
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ], p=0.7),
        transforms.ToTensor(),  # ✅ 要先變成 Tensor
        transforms.RandomErasing(p=0.3),  # ✅ 再做 Erasing 隨機遮住一部分
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
        'val': transforms.Compose([
            SquarePad(),
            transforms.Resize(resize),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

   

        # ---------- 自然排序用 ----------
    def natural_key(string):
        return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string)]

    # ---------- 建立正確的 class_to_idx ----------
    train_root = os.path.join(dir_path, 'train')
    sorted_class_dirs = sorted(os.listdir(train_root), key=natural_key)  # ✅ 自然排序 class 資料夾
    correct_class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted_class_dirs)}  # ✅ 對應正確順序

    # ---------- 建立 ImageFolder ----------
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(dir_path, x), data_transforms[x])
        for x in ['train', 'val']
    }

    # ---------- 套用正確的 class_to_idx ----------
    image_datasets['train'].class_to_idx = correct_class_to_idx
    image_datasets['val'].class_to_idx = correct_class_to_idx

    # ---------- 儲存 pkl 給 test.py 使用 ----------
    def save_obj(obj, dict_name):
        with open(dict_name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    save_obj(correct_class_to_idx, dictionary_name)

    # ---------- 輸出 class_to_idx 為 CSV ----------
    with open('./model/class_to_idx.csv', mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class_folder', 'label_index'])
        for cls_name, idx in correct_class_to_idx.items():
            writer.writerow([cls_name, idx])

    print("\n✅ 正確的 Class-to-Index Mapping：")
    for cls_name, idx in list(correct_class_to_idx.items())[:10]:  # 印前 10 個檢查
        print(f"Class Folder: {cls_name}  -->  Label Index: {idx}")
    print("✅ 已儲存 class_to_idx.pkl & .csv")

    

    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=6)
        for x in ['train', 'val']
    }
    print("\n📂 前幾張 val 的 label 對應：")
    for i in range(50):
        img, label = image_datasets['val'][i]
        print(f"Image {i}: Label = {label}, Class Folder = {list(correct_class_to_idx.keys())[list(correct_class_to_idx.values()).index(label)]}")

    print("Params to learn:")
    for name, param in model_ft.named_parameters():
        if param.requires_grad:
            print("\t", name)

    class_weights = compute_class_weights(image_datasets['train'])
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    model_ft, optimizer_ft, exp_lr_scheduler, start_epoch, best_acc = try_resume(
        model_ft, optimizer_ft, exp_lr_scheduler
    )

    model_ft, hist = train_model(
        model_ft, dataloaders_dict, criterion, optimizer_ft, exp_lr_scheduler,
        num_epochs, start_epoch=start_epoch, best_acc=best_acc
    )

