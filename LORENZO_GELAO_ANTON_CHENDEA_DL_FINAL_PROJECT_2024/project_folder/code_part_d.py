import copy
import os
import random
import sys

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


# Hyper Parameters
batch_size = 24
num_classes = 5  # 5 DR levels
learning_rate = 0.0001
num_epochs = 20


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


class CutOut(object):
    def __init__(self, mask_size, p=0.5):
        self.mask_size = mask_size
        self.p = p

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img

        # Ensure the image is a tensor
        if not isinstance(img, torch.Tensor):
            raise TypeError('Input image must be a torch.Tensor')

        # Get height and width of the image
        h, w = img.shape[1], img.shape[2]
        mask_size_half = self.mask_size // 2
        offset = 1 if self.mask_size % 2 == 0 else 0

        cx = np.random.randint(mask_size_half, w + offset - mask_size_half)
        cy = np.random.randint(mask_size_half, h + offset - mask_size_half)

        xmin, xmax = cx - mask_size_half, cx + mask_size_half + offset
        ymin, ymax = cy - mask_size_half, cy + mask_size_half + offset
        xmin, xmax = max(0, xmin), min(w, xmax)
        ymin, ymax = max(0, ymin), min(h, ymax)

        img[:, ymin:ymax, xmin:xmax] = 0
        return img


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

### NEW CODE HERE

def collect_predictions(models, dataloader):
    #This function is used for combining all models which are in the list "models", 
    #This function returns all_pred which is the combinaison of all predictions and all_labels which are the labels for each data    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.cpu().numpy()
            preds = []
            for model in models:
                outputs = model(images)
                preds.append(torch.softmax(outputs, dim=1).cpu().numpy())
            combined_preds = np.hstack(preds)
            all_preds.append(combined_preds)
            all_labels.append(labels)

    return np.vstack(all_preds), np.hstack(all_labels)

# We add theses implementations of the differents ensemble techniques we have to test here

# Stacking:

def ensemble_stacking(models, val_loader, train_loader, test_loader):
    # first we are using "collect_predictions" function to regroup predictions from all models
    all_val_preds, all_val_labels = collect_predictions(models, val_loader)
    all_train_preds, all_train_labels = collect_predictions(models, train_loader)

    # To train the stacking model on our predictions to learn to combine all predictions from every model in models , we use the logistic regression (maybe the most basic regression model)
    stacker = LogisticRegression()
    stacker.fit(all_train_preds, all_train_labels)
    
    stacker_preds = stacker.predict(all_val_preds)

    # printing of all metrics
    kappa, accuracy, precision, recall = compute_metrics(stacker_preds, all_val_labels)
    print(f"Stacking - Kappa: {kappa:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    #predicting and registering results for test dataset

    all_test_preds = []
    with torch.no_grad():
        for images in test_loader:
            if isinstance(images, (list, tuple)):
                images = images[0]
            images = images.to(device)
            preds = []
            for model in models_list:
                outputs = model(images)
                preds.append(torch.softmax(outputs, dim=1).cpu().numpy())
            combined_preds = np.hstack(preds)
            all_test_preds.append(combined_preds)

    all_test_preds = np.vstack(all_test_preds)

    stacker_test_preds = stacker.predict(all_test_preds)

    prediction_path='./test_predictions_part_d_stacker.csv'

    all_image_ids = []
    for i, data in enumerate(test_loader):
        images = data
        images = images.to(device)
        if not isinstance(images, list):
            image_ids = [os.path.basename(test_loader.dataset.data[idx]['img_path']) for idx in range(i * test_loader.batch_size, i * test_loader.batch_size + len(images))]
            all_image_ids.extend(image_ids)
    df = pd.DataFrame({
        'ID': all_image_ids,
        'TARGET': stacker_test_preds})
    df.to_csv(prediction_path, index=False)
    print(f'[Test] Save predictions to {os.path.abspath(prediction_path)}')

    return stacker

# Boosting

def ensemble_boosting(models, val_loader, train_loader, test_loader):
    # first we are using "collect_predictions" function to regroup predictions from all models
    all_val_preds, all_val_labels = collect_predictions(models, val_loader)
    all_train_preds, all_train_labels = collect_predictions(models, train_loader)
    
    # To train the boosting model on our predictions to learn to boost all predictions from every model in models with other predictions, 
    # we use the AdaBoostClassifier which is a basic Boosting model from sklearn library.
    boosting = AdaBoostClassifier(n_estimators=50)
    boosting.fit(all_train_preds, all_train_labels)
    
    boosting_preds = boosting.predict(all_val_preds)

    # printing of all metrics
    kappa, accuracy, precision, recall = compute_metrics(boosting_preds, all_val_labels)
    print(f"Boosting - Kappa: {kappa:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    #predicting and registering results for test dataset

    all_test_preds = []
    with torch.no_grad():
        for images in test_loader:
            if isinstance(images, (list, tuple)):
                images = images[0]
            images = images.to(device)
            preds = []
            for model in models_list:
                outputs = model(images)
                preds.append(torch.softmax(outputs, dim=1).cpu().numpy())
            combined_preds = np.hstack(preds)
            all_test_preds.append(combined_preds)

    all_test_preds = np.vstack(all_test_preds)

    boosting_test_preds = boosting.predict(all_test_preds)

    prediction_path='./test_predictions_part_d_boosting.csv'

    all_image_ids = []
    for i, data in enumerate(test_loader):
        images = data
        images = images.to(device)
        if not isinstance(images, list):
            image_ids = [os.path.basename(test_loader.dataset.data[idx]['img_path']) for idx in range(i * test_loader.batch_size, i * test_loader.batch_size + len(images))]
            all_image_ids.extend(image_ids)
    df = pd.DataFrame({
        'ID': all_image_ids,
        'TARGET': boosting_test_preds})
    df.to_csv(prediction_path, index=False)
    print(f'[Test] Save predictions to {os.path.abspath(prediction_path)}')

    return boosting

# Weighted average

def ensemble_weighted_average(models, val_loader, weights, test_loader):
    # first we are using "collect_predictions" function to regroup predictions from all models
    all_preds, all_labels = collect_predictions(models, val_loader)

    # Now we can calculate the weighted average
    # weight are based on the best kappa score for every model, a good indicator for the importance of every prediction

    #calculing the prediction matrix with every prediction by every model for every image
    pred_mat = []
    for element in all_preds:
        nb_models = len(element) // 24
        predictions = []
        for i in range(nb_models):
            predictions.append(np.argmax(element[i * 24 : (i + 1) * 24])) #selecting the class choosen by every model
        pred_mat.append(predictions) #adding the prediction array in the prediction matrix

    weighted_preds = np.round(np.average(np.array(pred_mat), axis=1, weights=weights)).astype(int) #prediction matrix must be an int array, so we rounding every value

    # printing of all metrics
    kappa, accuracy, precision, recall = compute_metrics(weighted_preds, all_labels)
    print(f"Weighted Average - Kappa: {kappa:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    #predicting and registering results for test dataset

    all_test_preds = []
    with torch.no_grad():
        for images in test_loader:
            if isinstance(images, (list, tuple)):
                images = images[0]
            images = images.to(device)
            preds = []
            for model in models_list:
                outputs = model(images)
                preds.append(torch.softmax(outputs, dim=1).cpu().numpy())
            combined_preds = np.hstack(preds)
            all_test_preds.append(combined_preds)

    all_test_preds = np.vstack(all_test_preds)

    prediction_path='./test_predictions_part_d_wa.csv'

    pred_test_mat = []
    for element in all_test_preds:
        nb_models = len(element) // 24
        predictions = []
        for i in range(nb_models):
            predictions.append(np.argmax(element[i * 24 : (i + 1) * 24])) #selecting the class choosen by every model
        pred_test_mat.append(predictions) #adding the prediction array in the prediction matrix

    weighted_preds_test = np.round(np.average(np.array(pred_test_mat), axis=1, weights=weights)).astype(int)

    all_image_ids = []
    for i, data in enumerate(test_loader):
        images = data
        images = images.to(device)
        if not isinstance(images, list):
            image_ids = [os.path.basename(test_loader.dataset.data[idx]['img_path']) for idx in range(i * test_loader.batch_size, i * test_loader.batch_size + len(images))]
            all_image_ids.extend(image_ids)
    df = pd.DataFrame({
        'ID': all_image_ids,
        'TARGET': weighted_preds_test})
    df.to_csv(prediction_path, index=False)
    print(f'[Test] Save predictions to {os.path.abspath(prediction_path)}')

    return weighted_preds

# Max voting

def ensemble_max_voting(models, val_loader, test_loader):
    # first we are using "collect_predictions" function to regroup predictions from all models
    all_preds, all_labels = collect_predictions(models, val_loader)

    # calculating of max voting, for every image, the function is choosing the majoritary prediction made by every model from models list
    max_voting_preds = []
    for element in all_preds:
        nb_models = len(element) // 24
        voting = []
        for i in range(nb_models):
            voting.append(np.argmax(element[i * 24 : (i + 1) * 24])) #selecting the class choosen by every model
        max_voting_preds.append(np.argmax(np.bincount(np.array(voting)))) #selecting the most choosen class
    
    max_voting_preds = np.array(max_voting_preds)

    # printing of all metrics
    kappa, accuracy, precision, recall = compute_metrics(max_voting_preds, all_labels)
    print(f"Max Voting - Kappa: {kappa:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    #predicting and registering results for test dataset

    all_test_preds = []
    with torch.no_grad():
        for images in test_loader:
            if isinstance(images, (list, tuple)):
                images = images[0]
            images = images.to(device)
            preds = []
            for model in models_list:
                outputs = model(images)
                preds.append(torch.softmax(outputs, dim=1).cpu().numpy())
            combined_preds = np.hstack(preds)
            all_test_preds.append(combined_preds)

    all_test_preds = np.vstack(all_test_preds)
    
    prediction_path='./test_predictions_part_d_maxvoting.csv'

    max_voting_preds_test = []
    for element in all_preds:
        nb_models = len(element) // 24
        voting = []
        for i in range(nb_models):
            voting.append(np.argmax(element[i * 24 : (i + 1) * 24])) #selecting the class choosen by every model
        max_voting_preds_test.append(np.argmax(np.bincount(np.array(voting)))) #selecting the most choosen class
    
    max_voting_preds_test = np.array(max_voting_preds_test)

    all_image_ids = []
    for i, data in enumerate(test_loader):
        images = data
        images = images.to(device)
        if not isinstance(images, list):
            image_ids = [os.path.basename(test_loader.dataset.data[idx]['img_path']) for idx in range(i * test_loader.batch_size, i * test_loader.batch_size + len(images))]
            all_image_ids.extend(image_ids)
    df = pd.DataFrame({
        'ID': all_image_ids,
        'TARGET': max_voting_preds_test})
    df.to_csv(prediction_path, index=False)
    print(f'[Test] Save predictions to {os.path.abspath(prediction_path)}')

    return max_voting_preds

# Bagging

def ensemble_bagging(models, val_loader, train_loader, test_loader):
    # first we are using "collect_predictions" function to regroup predictions from all models
    all_val_preds, all_val_labels = collect_predictions(models, val_loader)
    all_train_preds, all_train_labels = collect_predictions(models, train_loader)

    # To train the boosting model on our predictions to learn to aggregate all predictions from every model in models,
    # we are using BaggingClassifier which is the basic bagging classifier from sklearn librairy
    bagging = BaggingClassifier(n_estimators=50)
    bagging.fit(all_train_preds, all_train_labels)
    
    #predicting values for validation dataset
    bagging_preds = bagging.predict(all_val_preds)
    
    # printing of all metrics
    kappa, accuracy, precision, recall = compute_metrics(bagging_preds, all_val_labels)
    print(f"Bagging - Kappa: {kappa:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    #predicting and registering results for test dataset

    all_test_preds = []
    with torch.no_grad():
        for images in test_loader:
            if isinstance(images, (list, tuple)):
                images = images[0]
            images = images.to(device)
            preds = []
            for model in models_list:
                outputs = model(images)
                preds.append(torch.softmax(outputs, dim=1).cpu().numpy())
            combined_preds = np.hstack(preds)
            all_test_preds.append(combined_preds)

    all_test_preds = np.vstack(all_test_preds)

    bagging_test_preds = bagging.predict(all_test_preds)

    prediction_path='./test_predictions_part_d_bagging.csv'

    all_image_ids = []
    for i, data in enumerate(test_loader):
        images = data
        images = images.to(device)
        if not isinstance(images, list):
            image_ids = [os.path.basename(test_loader.dataset.data[idx]['img_path']) for idx in range(i * test_loader.batch_size, i * test_loader.batch_size + len(images))]
            all_image_ids.extend(image_ids)
    df = pd.DataFrame({
        'ID': all_image_ids,
        'TARGET': bagging_test_preds})
    df.to_csv(prediction_path, index=False)
    print(f'[Test] Save predictions to {os.path.abspath(prediction_path)}')

    return bagging

### NEW CODE ABOVE

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


def train_model(model, train_loader, val_loader, device, criterion, optimizer, lr_scheduler, num_epochs=25,
                checkpoint_path='model.pth'):
    best_model = model.state_dict()
    best_epoch = None
    best_val_kappa = -1.0  # Initialize the best kappa score

    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}/{num_epochs}')
        running_loss = []
        all_preds = []
        all_labels = []

        model.train()

        with tqdm(total=len(train_loader), desc=f'Training', unit=' batch', file=sys.stdout) as pbar:
            for images, labels in train_loader:
                if not isinstance(images, list):
                    images = images.to(device)  # single image case
                else:
                    images = [x.to(device) for x in images]  # dual images case

                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels.long())

                loss.backward()
                optimizer.step()

                preds = torch.argmax(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                running_loss.append(loss.item())

                pbar.set_postfix({'lr': f'{optimizer.param_groups[0]["lr"]:.1e}', 'Loss': f'{loss.item():.4f}'})
                pbar.update(1)

        lr_scheduler.step()

        epoch_loss = sum(running_loss) / len(running_loss)

        train_metrics = compute_metrics(all_preds, all_labels, per_class=True)
        kappa, accuracy, precision, recall = train_metrics[:4]

        print(f'[Train] Kappa: {kappa:.4f} Accuracy: {accuracy:.4f} '
              f'Precision: {precision:.4f} Recall: {recall:.4f} Loss: {epoch_loss:.4f}')

        if len(train_metrics) > 4:
            precision_per_class, recall_per_class = train_metrics[4:]
            for i, (precision, recall) in enumerate(zip(precision_per_class, recall_per_class)):
                print(f'[Train] Class {i}: Precision: {precision:.4f}, Recall: {recall:.4f}')

        # Evaluation on the validation set at the end of each epoch
        val_metrics = evaluate_model(model, val_loader, device)
        val_kappa, val_accuracy, val_precision, val_recall = val_metrics[:4]
        print(f'[Val] Kappa: {val_kappa:.4f} Accuracy: {val_accuracy:.4f} '
              f'Precision: {val_precision:.4f} Recall: {val_recall:.4f}')

        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            best_epoch = epoch
            best_model = model.state_dict()
            torch.save(best_model, checkpoint_path)

    print(f'[Val] Best kappa: {best_val_kappa:.4f}, Epoch {best_epoch}')

    return model


def evaluate_model(model, test_loader, device, test_only=False, prediction_path='./test_predictions.csv'):
    model.eval()

    all_preds = []
    all_labels = []
    all_image_ids = []

    with tqdm(total=len(test_loader), desc=f'Evaluating', unit=' batch', file=sys.stdout) as pbar:
        for i, data in enumerate(test_loader):

            if test_only:
                images = data
            else:
                images, labels = data

            if not isinstance(images, list):
                images = images.to(device)  # single image case
            else:
                images = [x.to(device) for x in images]  # dual images case

            with torch.no_grad():
                outputs = model(images)
                preds = torch.argmax(outputs, 1)

            if not isinstance(images, list):
                # single image case
                all_preds.extend(preds.cpu().numpy())
                image_ids = [
                    os.path.basename(test_loader.dataset.data[idx]['img_path']) for idx in
                    range(i * test_loader.batch_size, i * test_loader.batch_size + len(images))
                ]
                all_image_ids.extend(image_ids)
                if not test_only:
                    all_labels.extend(labels.numpy())
            else:
                # dual images case
                for k in range(2):
                    all_preds.extend(preds.cpu().numpy())
                    image_ids = [
                        os.path.basename(test_loader.dataset.data[idx][f'img_path{k + 1}']) for idx in
                        range(i * test_loader.batch_size, i * test_loader.batch_size + len(images[k]))
                    ]
                    all_image_ids.extend(image_ids)
                    if not test_only:
                        all_labels.extend(labels.numpy())

            pbar.update(1)

    # Save predictions to csv file for Kaggle online evaluation
    if test_only:
        df = pd.DataFrame({
            'ID': all_image_ids,
            'TARGET': all_preds
        })
        df.to_csv(prediction_path, index=False)
        print(f'[Test] Save predictions to {os.path.abspath(prediction_path)}')
    else:
        metrics = compute_metrics(all_preds, all_labels)
        return metrics


def compute_metrics(preds, labels, per_class=False):
    kappa = cohen_kappa_score(labels, preds, weights='quadratic')
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)

    # Calculate and print precision and recall for each class
    if per_class:
        precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
        recall_per_class = recall_score(labels, preds, average=None, zero_division=0)
        return kappa, accuracy, precision, recall, precision_per_class, recall_per_class

    return kappa, accuracy, precision, recall


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

        self.backbone = models.resnet34(pretrained=True)
        state_dict = torch.load('pretrained_DR_resize/pretrained/resnet34.pth', map_location='cpu')
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


class MyDualModel(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        backbone = models.resnet18(pretrained=True)
        backbone.fc = nn.Identity()

        # Here the two backbones will have the same structure but unshared weights
        self.backbone1 = copy.deepcopy(backbone)
        self.backbone2 = copy.deepcopy(backbone)

        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, images):
        image1, image2 = images

        x1 = self.backbone1(image1)
        x2 = self.backbone2(image2)

        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # Choose between 'single image' and 'dual images' pipeline
    # This will affect the model definition, dataset pipeline, training and evaluation

    mode = 'single'  # forward single image to the model each time
    # mode = 'dual'  # forward two images of the same eye to the model and fuse the features

    # Create datasets
    train_dataset = RetinopathyDataset('./DeepDRiD/train.csv', './DeepDRiD/train/', transform_train, mode)
    val_dataset = RetinopathyDataset('./DeepDRiD/val.csv', './DeepDRiD/val/', transform_test, mode)
    test_dataset = RetinopathyDataset('./DeepDRiD/test.csv', './DeepDRiD/test/', transform_test, mode, test=True)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define the weighted CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()

    # Use GPU device is possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # definition of the model, which is a combianaison of three different manualy pretrained model
    #calling of differents models
    
    #charging of the resnet model
    resnet34 = MyModel_resnet()
    resnet34 = resnet34.to(device)
    state_dict = torch.load('./model_b_resnet.pth', map_location=device)
    resnet34.load_state_dict(state_dict, strict=True)
    resnet34 = resnet34.to(device)
    resnet34.eval()

    #charging of the Efficientnet model
    efficientnet = MyModel_efficientnet()
    efficientnet = efficientnet.to(device)
    state_dict = torch.load('./model_b_efficientnet.pth', map_location=device)
    efficientnet.load_state_dict(state_dict, strict=True)
    efficientnet = efficientnet.to(device)
    efficientnet.eval()

    #charging of the densenet model
    densenet = MyModel_densenet()
    densenet = densenet.to(device)
    state_dict = torch.load('./model_b_densenet.pth', map_location=device)
    densenet.load_state_dict(state_dict, strict=True)
    densenet = densenet.to(device)
    densenet.eval()

    models_list = [resnet34, efficientnet, densenet]

    print(models_list, '\n')
    print('Pipeline Mode:', mode)

    stacking = ensemble_stacking(models_list, val_loader, train_loader, test_loader)
    boosting = ensemble_boosting(models_list, val_loader, train_loader, test_loader)
    weights = [0.2, 0.4, 0.4] #based on best kappas for every models
    wa = ensemble_weighted_average(models_list, val_loader, weights, test_loader)
    mv = ensemble_max_voting(models_list, val_loader, test_loader)
    bagging = ensemble_bagging(models_list, val_loader, train_loader, test_loader)

    #to generate a test_prediction.csv, we reuse the bagging method

    #first we have to collect all predictions, we can't use the collect_predictions function because we don't have any labels for test
    
    


