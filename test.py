import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ttach
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CustomDataset, BUSIDataset
from model import StackWiseIntegrationModel
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from PIL import Image


def plot_confusion_matrix(conf_matrix, class_names, accuracy, f1, output_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Normalized Confusion Matrix\nAccuracy: {accuracy:.4f}\nF1 Score: {f1:.4f}')
    
    plt.savefig(output_path)
    plt.close()


def load_model(model_path, device, mode):
    model = StackWiseIntegrationModel(mode=mode)
            
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # TTA 적용
    # model = ttach.ClassificationTTAWrapper(model, ttach_transform)
    
    return model


def test_model(model, test_loader, device, output_path):
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for b_mode_images, se_mode_images, labels in tqdm(test_loader, desc="Testing"):
            b_mode_images, se_mode_images, labels = b_mode_images.to(device), se_mode_images.to(device), labels.to(device)
            
            outputs = model(b_mode_images, se_mode_images)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='macro')
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    class_names = [str(i) for i in range(1, 1+len(test_loader.dataset.classes))]
    
    print('Confusion Matrix:')
    print(conf_matrix, '\n')
    
    print('Normalized Confusion Matrix:')
    print(conf_matrix_normalized, '\n')
    
    print('Classification Report:')
    print(classification_report(all_labels, all_predictions, target_names=class_names))

    plot_confusion_matrix(conf_matrix_normalized, class_names, accuracy, f1, output_path)
    

test_transform = transforms.Compose([
    transforms.Resize((448, 448), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

ttach_transform = ttach.Compose([
    ttach.HorizontalFlip(),
    # ttach.FiveCrops(int(384*0.8), int(384*0.8))
])


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_data_dir = 'data/test'
    batch_size = 16
    mode = 'b_mode'  # 'both', 'b_mode', 'se_mode' 중 선택
    
    test_dataset = BUSIDataset(root_dir=test_data_dir, transform=test_transform)
    # test_dataset = CustomDataset(root_dir=test_data_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    pt_files = [f for f in os.listdir() if f.endswith('.pt')]
    for model_path in pt_files:
        print(f"Testing model: {model_path}")
        output_path = f"{model_path.split('.pt')[0]}.png"
        model = load_model(model_path, device, mode)
        test_model(model, test_loader, device, output_path)