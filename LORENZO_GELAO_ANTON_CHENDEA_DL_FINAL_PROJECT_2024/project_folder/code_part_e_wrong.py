import copy
import os
import random
import sys
import cv2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageDraw 
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from torchvision import models

# Hyper Parameters
batch_size = 24
epochs = 20

# CHANGE PARAMETERS
# Choose the path for the model
path = './model_a_resnet.pth'
# IMPLEMENTATION, DON'T CHANGE

#class implementations

class RetinopathyDataset(Dataset):
    def __init__(self, ann_file, image_dir, transform=None, mode='single', test=False):
        self.ann_file = ann_file
        self.image_dir = image_dir
        self.transform = transform

        self.test = test
        self.mode = mode

        if self.mode == 'single':
            self.data = self.load_data()
        else:
            self.data = self.load_data_dual()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode == 'single':
            return self.get_item(index)
        else:
            return self.get_item_dual(index)

    # 1. single image
    def load_data(self):
        df = pd.read_csv(self.ann_file)

        data = []
        for _, row in df.iterrows():
            file_info = dict()
            file_info['img_path'] = os.path.join(self.image_dir, row['img_path'])
            if not self.test:
                file_info['dr_level'] = int(row['patient_DR_Level'])
            data.append(file_info)
        return data

    def get_item(self, index):
        data = self.data[index]
        img = Image.open(data['img_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)

        if not self.test:
            label = torch.tensor(data['dr_level'], dtype=torch.int64)
            return img, label
        else:
            return img

    # 2. dual image
    def load_data_dual(self):
        df = pd.read_csv(self.ann_file)

        df['prefix'] = df['image_id'].str.split('_').str[0]  # The patient id of each image
        df['suffix'] = df['image_id'].str.split('_').str[1].str[0]  # The left or right eye
        grouped = df.groupby(['prefix', 'suffix'])

        data = []
        for (prefix, suffix), group in grouped:
            file_info = dict()
            file_info['img_path1'] = os.path.join(self.image_dir, group.iloc[0]['img_path'])
            file_info['img_path2'] = os.path.join(self.image_dir, group.iloc[1]['img_path'])
            if not self.test:
                file_info['dr_level'] = int(group.iloc[0]['patient_DR_Level'])
            data.append(file_info)
        return data

    def get_item_dual(self, index):
        data = self.data[index]
        img1 = Image.open(data['img_path1']).convert('RGB')
        img2 = Image.open(data['img_path2']).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if not self.test:
            label = torch.tensor(data['dr_level'], dtype=torch.int64)
            return [img1, img2], label
        else:
            return [img1, img2]

class MyModel_densenet(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        self.backbone = models.densenet121(pretrained=True)
        state_dict = torch.load('pretrained_DR_resize/pretrained/densenet121.pth', map_location='cpu')
        info = self.backbone.load_state_dict(state_dict, strict=False)
        print('missing keys:', info[0])
        print('unexpected keys:', info[1])

        self.fc = nn.Sequential(
            nn.Linear(1000, 24),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(24, 24),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(24, num_classes ** 2 - 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x
    
class MyModel_resnet(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove the original classification layer

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

class MyModel_efficientnet(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        self.backbone = models.efficientnet_b0(pretrained=True)
        state_dict = torch.load('pretrained_DR_resize/pretrained/efficientnet_b0.pth', map_location='cpu')
        info = self.backbone.load_state_dict(state_dict, strict=False)
        print('missing keys:', info[0])
        print('unexpected keys:', info[1])

        self.fc = nn.Sequential(
            nn.Linear(1000, 24),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(24, 24),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(24, num_classes ** 2 - 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x
    
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        self.target_layer.register_backward_hook(self.save_gradient)

    def save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def __call__(self, x, class_idx=None):
        outputs = []
        def hook(module, input, output):
            outputs.append(output)
        handle = self.target_layer.register_forward_hook(hook)

        logits = self.model(x)
        handle.remove()

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[:, class_idx].backward(retain_graph=True)

        gradients = self.gradients[0].cpu().detach().numpy()
        activations = outputs[0][0].cpu().detach().numpy()

        weights = gradients.mean(axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cam / cam.max()
        return cam
    
class SLORandomPad:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        pad_width = max(0, self.size[0] - img.width)
        pad_height = max(0, self.size[1] - img.height)
        pad_left = random.randint(0, pad_width)
        pad_top = random.randint(0, pad_height)
        pad_right = pad_width - pad_left
        pad_bottom = pad_height - pad_top
        return transforms.functional.pad(img, (pad_left, pad_top, pad_right, pad_bottom))


class FundRandomRotate:
    def __init__(self, prob, degree):
        self.prob = prob
        self.degree = degree

    def __call__(self, img):
        if random.random() < self.prob:
            angle = random.uniform(-self.degree, self.degree)
            return transforms.functional.rotate(img, angle)
        return img
    
transform_train = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomCrop((110, 110)),
    SLORandomPad((224, 224)),
    FundRandomRotate(prob=0.5, degree=30),
    transforms.RandomHorizontalFlip(p=0.35),
    transforms.RandomVerticalFlip(p=0.35),
    transforms.ColorJitter(brightness=(0.1, 0.9)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def evaluate_finetuned_model(num_epochs, model, train_loader, val_loader, device, criterion):
    #Evaluating a finetuned model by epochs
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.25)

    model.eval()
    with tqdm(total=num_epochs, desc=f'Evaluating', unit='epochs', file=sys.stdout) as pbar:
        # Evaluation by epochs
        for epoch in range(num_epochs):
            running_train_loss = 0.0
            correct_train = 0
            total_train = 0
            with torch.no_grad():
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    running_train_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total_train += labels.size(0)
                    correct_train += (predicted == labels).sum().item()

            train_loss = running_train_loss / len(train_loader)
            train_accuracy = correct_train / total_train
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            running_val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    running_val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
            pbar.update(1)

        val_loss = running_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

    return train_losses, train_accuracies, val_losses, val_accuracies



# main code part
if __name__ == '__main__':
    # Choose between 'single image' and 'dual images' pipeline
    # This will affect the model definition, dataset pipeline, training and evaluation

    mode = 'single'  # forward single image to the model each time
    device = "cuda"

    # calling the fine tuned model
    if "resnet" in path:
        model = MyModel_resnet()
        target_layer = model.backbone.layer4
    elif "densenet" in path:
        model = MyModel_densenet()
        target_layer = model.backbone.features
    elif "efficientnet" in path:
        model = MyModel_efficientnet()
        target_layer = model.backbone.features[-1]
    else:
        raise ValueError("Input error : bad path name")

    model = model.to(device)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    train_dataset = RetinopathyDataset('./DeepDRiD/train.csv', './DeepDRiD/train/', transform_train, mode)
    val_dataset = RetinopathyDataset('./DeepDRiD/val.csv', './DeepDRiD/val/', transform_test, mode)
    test_dataset = RetinopathyDataset('./DeepDRiD/test.csv', './DeepDRiD/test/', transform_test, mode, test=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    train_losses, train_accuracies, val_losses, val_accuracies  = evaluate_finetuned_model(epochs, model, train_loader, val_loader, device, criterion)

    print(train_losses, train_accuracies)
    print(val_losses, val_accuracies)

    x_train = range(1, len(train_losses) + 1)
    x_val = range(1, len(val_losses) + 1)

    # Plotting
    plt.figure(figsize=(12, 5))

    # Losses 
    plt.subplot(1, 2, 1)
    plt.plot(x_train, train_losses, label="Train Loss", marker='o')
    plt.plot(x_val, val_losses, label="Validation Loss", marker='o')
    plt.title("Train vs Validation Loss")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    # Accuracies
    plt.subplot(1, 2, 2)
    plt.plot(x_train, train_accuracies, label="Train Accuracy", marker='o')
    plt.plot(x_val, val_accuracies, label="Validation Accuracy", marker='o')
    plt.title("Train vs Validation Accuracy")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
