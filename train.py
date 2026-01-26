import os
import torch
import torch.optim as optim
import math
import numpy as np
import random
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from dataset import CustomDataset, BUSIDataset
from model import StackWiseIntegrationModel
from tqdm import tqdm
from sklearn.model_selection import KFold
from PIL import Image


def seed_torch(seed=1):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
            

def split_dataset_kfold(dataset, n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = list(range(len(dataset)))
    
    return kf.split(indices)


def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=10, fold=1, use_scheduler=True):
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = -1
    
    for epoch in range(1, 1+num_epochs):
        print(f"(Fold {fold}) Epoch [{epoch}/{num_epochs}]")
        # Train
        for param_group in optimizer.param_groups:
            print(f"Learning rate: {param_group['lr']:.5f}")
        model.train()
        train_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Train")
        
        for b_mode_images, se_mode_images, labels in train_loader_tqdm:
            b_mode_images, se_mode_images, labels = b_mode_images.to(device), se_mode_images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(b_mode_images, se_mode_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_mode_images.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for b_mode_images, se_mode_images, labels in tqdm(val_loader, desc="Validation"):
                b_mode_images, se_mode_images, labels = b_mode_images.to(device), se_mode_images.to(device), labels.to(device)
                outputs = model(b_mode_images, se_mode_images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * b_mode_images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader.dataset)
        val_accuracy = correct / len(val_loader.dataset)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Model save (At the lowest validation loss)
        if val_loss < best_val_loss:
            if best_epoch != -1:
                os.remove(f'fold_{fold}_{best_epoch}_{best_val_loss:.4f}.pt')
            else:
                pass
            
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            best_epoch = epoch
            
            torch.save(best_model_state, f'fold_{fold}_{best_epoch}_{best_val_loss:.4f}.pt')
            print(f"Lowest validation loss: {best_val_loss:.4f} in (Fold {fold}).\n")

        if use_scheduler:
            scheduler.step()
    
    return best_val_loss, best_model_state, best_epoch


# Data Augmentation
train_transform = transforms.Compose([
    transforms.Resize((448, 448), interpolation=Image.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    # transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((448, 448), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


if __name__ == "__main__":
    seed_torch(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = 'data/train'
    n_splits = 5  # Number of fold (K-Fold Cross Validation)
    batch_size = 32
    
    num_epochs = 300
    learning_rate = 0.0002
    warmup_epochs = 10
    warmup_decay = 0.01
    min_lr = 1e-6
    use_scheduler = True  # 학습률 스케줄러 사용 여부
    
    mode = 'b_mode'  # 'both', 'b_mode', 'se_mode'

    full_dataset = BUSIDataset(root_dir=data_dir, transform=None)
    # full_dataset = CustomDataset(root_dir=data_dir, transform=None)
    kf = split_dataset_kfold(full_dataset, n_splits=n_splits, seed=42)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
    
    all_val_losses = []
    best_model_state = None
    best_val_loss = float('inf')
    best_fold = -1
    best_epoch = -1
    
    for fold, (train_idx, val_idx) in enumerate(kf, start=1):
        print(f"\n================Fold {fold}================\n")
        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_dataset, val_idx)
        
        train_subset.dataset.transform = train_transform
        val_subset.dataset.transform = val_transform

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        model = StackWiseIntegrationModel(mode=mode)
        
        model.to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        
        if use_scheduler:
            lr_lambda = lambda epoch: (epoch * (1 - warmup_decay) / warmup_epochs + warmup_decay) \
            if epoch < warmup_epochs else \
            (1 - min_lr / learning_rate) * 0.5 * (math.cos((epoch - warmup_epochs) / (num_epochs - warmup_epochs) * math.pi) + 1) + min_lr / learning_rate
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
        else:
            scheduler = None
        
        val_loss, model_state, epoch = train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, fold, use_scheduler)
        all_val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model_state
            best_fold = fold
            best_epoch = epoch
            
    average_val_loss = np.mean(all_val_losses)

    print()
    print(f'Average Validation Loss: {average_val_loss:.4f}')
    print(f'Lowest validation loss: {best_val_loss:.4f} at (Fold {best_fold}) Epoch {best_epoch}')
    
    # Save configuration
    config_text = f"""transform = {train_transform}
    n_splits = {n_splits}
    batch_size = {batch_size}
    
    num_epochs = {num_epochs}
    learning_rate = {learning_rate}
    warmup_epochs = {warmup_epochs}
    warmup_decay = {warmup_decay}
    min_lr = {min_lr}
    use_scheduler = {use_scheduler}
    
    model = {model.model_ft.__class__.__name__}
    mode = {mode}
    """
    
    final_output = f"""
    Average Validation Loss: {average_val_loss:.4f}
    Lowest validation loss: {best_val_loss:.4f} at (Fold {best_fold}) Epoch {best_epoch}
    """

    with open('config.txt', 'w') as f:
        f.write(config_text)
        f.write(final_output)