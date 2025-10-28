import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
from torchvision import models
from sklearn.model_selection import KFold  # 新增 KFold 用于 K折交叉验证
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import numpy as np
import random
from model import resnet_cbam_reg
from Dataset_signal import NumpySignalDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report
)

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

def set_random_seed(seed=42):
    """确保可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Evaluate function
def evaluate_on_loader(model, data_loader, device):
    model.eval()
    all_labels = []
    all_scores = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            pos_probs = probs[:, 1]
            all_labels.append(labels.cpu().numpy())
            all_scores.append(pos_probs.cpu().numpy())
    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)
    return all_labels, all_scores


# Training and evaluation function for each fold
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=25):
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation loop
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item()

        val_loss = running_val_loss / len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}')

    return model, best_model_wts, train_losses, val_losses


# Main training and evaluation function with KFold
def main(args):
    set_random_seed(args.seed)

    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter() if args.display else None

    gpus = args.gpu.split(',')
    is_use_cuda = torch.cuda.is_available()
    cudnn.benchmark = True
    device = torch.device('cuda' if is_use_cuda else 'cpu')

    # Load dataset
    full_dataset = NumpySignalDataset(
        data_dir=args.data_dir,
        label_np_dir=args.label_np,
        label_p_dir=args.label_p
    )

    # K-Fold Cross Validation
    kf = KFold(n_splits=args.k_folds, shuffle=False)

    all_fold_losses = []
    all_fold_models = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        print(f"Fold {fold + 1}/{args.k_folds}")

        print(f"Train indices for fold {fold + 1}: {train_idx}")
        print(f"Validation indices for fold {fold + 1}: {val_idx}")

        # Create data loaders for this fold
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # Initialize model for this fold
        model = resnet_cbam_reg.resnet18_cbam(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        lr_schedule = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)

        early_stopping = EarlyStopping(patience=10, min_delta=0.001)

        # Train and validate the model for this fold
        model, best_model_wts, train_losses, val_losses = train_model(
            model, train_loader, val_loader, criterion, optimizer, device, num_epochs=args.num_epochs
        )

        # Save the best model weights for this fold
        torch.save(best_model_wts, f'model_fold_{fold + 1}.pth')
        all_fold_models.append(best_model_wts)
        all_fold_losses.append((train_losses, val_losses))

        # Update learning rate schedule
        lr_schedule.step()

    # Optionally, save the final aggregated model after K-Fold
    final_model = resnet_cbam_reg.resnet18_cbam(pretrained=False)
    final_model.fc = nn.Linear(final_model.fc.in_features, 2)
    final_model.to(device)
    final_model.load_state_dict(all_fold_models[0])  # Placeholder for aggregating fold models if necessary
    torch.save(final_model.state_dict(), 'final_model.pth')

    print("Training and evaluation completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNet18 Training & Evaluation with K-Fold Cross Validation')
    parser.add_argument('--data_dir', type=str,
                        default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_Signal_CWT_pt', help='Path to training data')
    parser.add_argument('--label_np', type=str,
                        default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\original_elementValue_npyFiles\damageM_Data\NP',
                        help='Path to NP label folder')
    parser.add_argument('--label_p', type=str,
                        default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\original_elementValue_npyFiles\damageM_Data\P',
                        help='Path to P label folder')
    parser.add_argument('--gpu', default='cuda:0', type=str, help='GPU ID(s) to use')
    parser.add_argument('--model', default='resnet18_cbam', type=str, help='Model name used for logging')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size per GPU')
    parser.add_argument('--display', action='store_true', help='Use Tensorboard for visualization')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--k_folds', type=int, default=10, help='Number of folds for cross-validation')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
    args = parser.parse_args()
    main(args)
