import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score
from torchvision.transforms import v2
from tqdm import tqdm

from src.FixedSkinModel import FixedSkinModel
from config import CONDITIONS, MODEL_PATH

# ---------------------------------------------------------
# Transforms & Dataset
# ---------------------------------------------------------
def get_transforms(is_train=True):
    if is_train:
        return v2.Compose([
            v2.ToImage(),
            v2.RandomResizedCrop(224, scale=(0.8, 1.0)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.1),
            v2.RandomRotation(degrees=15),
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            v2.RandomGrayscale(p=0.05),
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            v2.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            v2.RandomErasing(p=0.1)
        ])
    else:
        return v2.Compose([
            v2.ToImage(),
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class SkinDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        labels = torch.tensor([row[c] for c in CONDITIONS], dtype=torch.float32)
        return image, labels

# ---------------------------------------------------------
# Class Imbalance, Mixup & Early Stopping
# ---------------------------------------------------------
def compute_pos_weights(df, conditions):
    pos_weights = []
    for condition in conditions:
        pos = df[condition].sum()
        neg = len(df) - pos
        weight = neg / max(pos, 1)
        pos_weights.append(min(weight, 10.0))
    return torch.tensor(pos_weights, dtype=torch.float32)

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == 'max' and score < self.best_score + self.min_delta) or \
             (self.mode == 'min' and score > self.best_score - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0

# ---------------------------------------------------------
# Optimal Thresholds Finder
# ---------------------------------------------------------
def find_optimal_thresholds(model, val_loader, device, conditions):
    """ Sweeps thresholds from 0.1 to 0.9 in steps of 0.01 per condition, picks threshold maximizing F1, saves result """
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            all_probs.append(torch.sigmoid(logits).cpu())
            all_targets.append(targets.cpu())
            
    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    thresholds_dict = {}
    for i, condition in enumerate(conditions):
        best_f1 = 0
        best_t = 0.5
        for t in np.arange(0.1, 0.91, 0.01):
            preds = (all_probs[:, i] >= t).astype(int)
            f1 = f1_score(all_targets[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        thresholds_dict[condition] = best_t
        
    os.makedirs('models', exist_ok=True)
    with open('models/optimal_thresholds.json', 'w') as f:
        json.dump(thresholds_dict, f, indent=4)
        
    # Attempt to inject it automatically into the checkpoint
    if os.path.exists('models/fixed_skin_model.pth'):
        try:
            ckpt = torch.load('models/fixed_skin_model.pth', map_location='cpu')
            ckpt['optimal_thresholds'] = thresholds_dict
            torch.save(ckpt, 'models/fixed_skin_model.pth')
        except:
             pass

    return thresholds_dict

def evaluate_on_test(model, test_loader, device, conditions, thresholds):
    """ Loads best checkpoint, runs inference using optimal thresholds, prints full classification report """
    checkpoint = torch.load('models/fixed_skin_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            all_probs.append(torch.sigmoid(logits).cpu())
            all_targets.append(targets.cpu())
            
    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    all_preds = np.zeros_like(all_probs)
    for i, c in enumerate(conditions):
        t = thresholds.get(c, 0.5)
        all_preds[:, i] = (all_probs[:, i] >= t).astype(int)
        
    from sklearn.metrics import classification_report, confusion_matrix
    
    report_lines = []
    for i, c in enumerate(conditions):
        rep = classification_report(all_targets[:, i], all_preds[:, i], zero_division=0)
        cm = confusion_matrix(all_targets[:, i], all_preds[:, i])
        
        info = f"--- {c.upper()} ---\nClassification Report:\n{rep}\nConfusion Matrix:\n{cm}\n\n"
        print(info)
        report_lines.append(info)
        
    with open('models/training_report.txt', 'w') as f:
        f.write("".join(report_lines))
    print("✅ Full test evaluation saved to models/training_report.txt")

# ---------------------------------------------------------
# Training Function
# ---------------------------------------------------------
def train_fixed_model(train_df, val_df, test_df, epochs=50, batch_size=16, skip_phase1=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('models', exist_ok=True)
    
    train_dataset = SkinDataset(train_df, transform=get_transforms(is_train=True))
    val_dataset = SkinDataset(val_df, transform=get_transforms(is_train=False))
    test_dataset = SkinDataset(test_df, transform=get_transforms(is_train=False))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = FixedSkinModel(num_classes=8).to(device)
    pos_weight = compute_pos_weights(train_df, CONDITIONS).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    early_stopping = EarlyStopping(patience=10, min_delta=0.001, mode='max')
    
    train_losses, val_losses, val_f1_history = [], [], []
    best_val_f1 = 0.0
    
    phase = "Head Only"
    if skip_phase1:
        model.unfreeze_backbone(num_layers=30)
        phase = "Fine-tuning"
        
        backbone_params, head_params = [], []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'backbone.features' in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)
        optimizer = optim.AdamW([{'params': backbone_params, 'lr': 1e-4}, {'params': head_params, 'lr': 5e-4}], weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    else:
        model.freeze_backbone()
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=10, steps_per_epoch=len(train_loader), pct_start=0.3)
    
    for epoch in tqdm(range(1, epochs + 1), desc="Training Epochs"):
        if not skip_phase1 and epoch == 11:
            phase = "Fine-tuning"
            model.unfreeze_backbone(num_layers=30)
            backbone_params, head_params = [], []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if 'backbone.features' in name:
                        backbone_params.append(param)
                    else:
                        head_params.append(param)
            optimizer = optim.AdamW([{'params': backbone_params, 'lr': 1e-4}, {'params': head_params, 'lr': 5e-4}], weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
            
        epoch_lr = optimizer.param_groups[-1]['lr']
        
        # Train Loop
        model.train()
        running_loss = 0.0
        all_train_preds, all_train_targets = [], []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            applied_mixup = False
            
            if np.random.rand() < 0.3:
                applied_mixup = True
                
            if applied_mixup:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.2)
                optimizer.zero_grad()
                logits = model(inputs)
                loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
            else:
                optimizer.zero_grad()
                logits = model(inputs)
                loss = criterion(logits, targets)
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if phase == "Head Only":
                scheduler.step()
                
            running_loss += loss.item()
            
            if not applied_mixup:
                preds = (torch.sigmoid(logits) > 0.5).float()
                all_train_preds.append(preds.cpu())
                all_train_targets.append(targets.cpu())
                
        if phase == "Fine-tuning":
            scheduler.step()
            
        train_loss = running_loss / len(train_loader)
        if len(all_train_preds) > 0:
            tp = torch.cat(all_train_preds).numpy()
            tt = torch.cat(all_train_targets).numpy()
            train_f1 = f1_score(tt, tp, average='macro', zero_division=0)
            train_acc = accuracy_score(tt, tp)
        else:
            train_f1 = train_acc = 0.0
            
        # Val Loop
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                loss = criterion(logits, targets)
                val_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())
                
        val_loss /= len(val_loader)
        tp = torch.cat(all_preds).numpy()
        tt = torch.cat(all_targets).numpy()
        val_f1 = f1_score(tt, tp, average='macro', zero_division=0)
        val_acc = accuracy_score(tt, tp)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1_history.append(val_f1)
        
        early_stopping(val_f1)
        is_best = False
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            is_best = True
            
        es_status = f"{early_stopping.counter}/{early_stopping.patience}"
        
        # Table output
        print(f"\n┌{'─'*65}┐")
        print(f"│ Epoch {epoch:02d}/{epochs:02d} | Phase: {phase:<11} | LR: {epoch_lr:.6f}               │")
        print(f"├{'─'*10}┬{'─'*12}┬{'─'*12}┬{'─'*12}┬{'─'*14}┤")
        print(f"│ Loss     │ {train_loss:<10.4f} │ {val_loss:<10.4f} │ {'-':<10} │ {es_status:<12} │")
        print(f"│ F1 Macro │ {train_f1:<10.4f} │ {val_f1:<10.4f} │ {best_val_f1:<10.4f} │              │")
        print(f"│ Accuracy │ {train_acc*100:<9.2f}% │ {val_acc*100:<9.2f}% │ {'-':<10} │              │")
        print(f"└{'─'*10}┴{'─'*12}┴{'─'*12}┴{'─'*12}┴{'─'*14}┘")
        
        if is_best:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'optimal_thresholds': {}, # appended later
                'conditions': CONDITIONS,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_f1_history': val_f1_history
            }, MODEL_PATH)
            print("🌟 Saved new best model checkpoint!")
            
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, 'models/last_epoch.pth')
        
        if early_stopping.should_stop:
            print(f"🛑 Early stopping triggered at epoch {epoch}")
            break
            
    return model, val_loader, test_loader, device