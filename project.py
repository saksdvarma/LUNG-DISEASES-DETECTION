import os
import sys
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class Config:
    """Configuration class for hyperparameters and paths"""
    # Data
    DATA_DIR = Path('./data')
    OUTPUT_DIR = Path('./outputs')
    KAGGLE_DATASET = 'tawsifurrahman/covid19-radiography-database'

    # Model
    IMG_SIZE = 224
    BATCH_SIZE = 16
    NUM_CLASSES = 3
    NUM_EPOCHS = 25
    LEARNING_RATE = 0.001
    LEARNING_RATE_FINETUNE = 0.0001

    # Training
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4

    # Class names
    CLASS_NAMES = ['COVID', 'Normal', 'Viral Pneumonia']

    # Augmentation parameters
    ROTATION_RANGE = 15
    ZOOM_RANGE = 0.15
    SHIFT_RANGE = 0.1

    def __init__(self):
        self.DATA_DIR.mkdir(exist_ok=True)
        self.OUTPUT_DIR.mkdir(exist_ok=True)


class TraditionalCNN(nn.Module):
    """Traditional CNN architecture as baseline"""

    def __init__(self, num_classes=3):
        super(TraditionalCNN, self).__init__()

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )

        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )

        # For 224x224 input: 224 -> 112 -> 56 -> 28 -> 14
        self.flatten_size = 256 * 14 * 14

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.fc(x)
        return x


def create_resnet50(num_classes=3, pretrained=True):
    """Create ResNet50 model with transfer learning"""
    model = models.resnet50(pretrained=pretrained)

    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    return model


def create_efficientnet(num_classes=3, pretrained=True):
    """Create EfficientNetB0 model with transfer learning"""
    try:
        import timm
        model = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0)

        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model.num_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        return model

    except ImportError:
        print("Warning: timm not installed. Using torchvision EfficientNet")
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(pretrained=pretrained)

        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        return model


class DatasetDownloader:
    """Download and organize COVID-19 dataset"""

    @staticmethod
    def download_kaggle_dataset(dataset_name: str, output_dir: Path):
        """Download dataset from Kaggle"""
        print(f"Downloading {dataset_name}...")

        kaggle_dir = Path.home() / '.kaggle'
        if not (kaggle_dir / 'kaggle.json').exists():
            print("\n⚠️  Kaggle API key not found!")
            print("1. Go to https://www.kaggle.com/settings/account")
            print("2. Click 'Create New API Token'")
            print("3. Place kaggle.json in ~/.kaggle/")
            print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
            sys.exit(1)

        os.system(f"kaggle datasets download -d {dataset_name}")

        zip_file = dataset_name.split('/')[-1] + '.zip'
        print(f"Extracting {zip_file}...")
        import zipfile
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

        os.remove(zip_file)
        print("✓ Dataset downloaded and extracted!")

    @staticmethod
    def organize_dataset(source_dir: Path, target_dir: Path, split_ratios=(0.7, 0.15, 0.15)):
        """Organize dataset into train/val/test splits"""
        print("\nOrganizing dataset...")

        covid_images = source_dir / 'COVID-19_Radiography_Dataset' / 'COVID' / 'images'
        normal_images = source_dir / 'COVID-19_Radiography_Dataset' / 'Normal' / 'images'
        pneumonia_images = source_dir / 'COVID-19_Radiography_Dataset' / 'Viral Pneumonia' / 'images'

        for split in ['train', 'val', 'test']:
            for class_name in ['COVID', 'Normal', 'Viral_Pneumonia']:
                (target_dir / split / class_name).mkdir(parents=True, exist_ok=True)

        class_mapping = {
            'COVID': covid_images,
            'Normal': normal_images,
            'Viral_Pneumonia': pneumonia_images
        }

        stats = {}
        for class_name, class_path in class_mapping.items():
            if not class_path.exists():
                print(f"Warning: {class_path} not found, skipping...")
                continue

            images = list(class_path.glob('*.png'))
            np.random.shuffle(images)

            n_train = int(len(images) * split_ratios[0])
            n_val = int(len(images) * split_ratios[1])

            train_imgs = images[:n_train]
            val_imgs = images[n_train:n_train + n_val]
            test_imgs = images[n_train + n_val:]

            for img in train_imgs:
                shutil.copy(img, target_dir / 'train' / class_name / img.name)
            for img in val_imgs:
                shutil.copy(img, target_dir / 'val' / class_name / img.name)
            for img in test_imgs:
                shutil.copy(img, target_dir / 'test' / class_name / img.name)

            stats[class_name] = {
                'train': len(train_imgs),
                'val': len(val_imgs),
                'test': len(test_imgs),
                'total': len(images)
            }

            print(f"{class_name}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")

        return stats


def get_transforms(config: Config, augment=False):
    """Get data transforms"""
    if augment:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomRotation(config.ROTATION_RANGE),
            transforms.RandomAffine(
                degrees=0,
                translate=(config.SHIFT_RANGE, config.SHIFT_RANGE),
                scale=(1 - config.ZOOM_RANGE, 1 + config.ZOOM_RANGE)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    predictions = []
    targets = []

    pbar = tqdm(dataloader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
        targets.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(targets, predictions)

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    predictions = []
    targets = []
    probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            probabilities = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            probs.extend(probabilities.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(targets, predictions)

    return epoch_loss, epoch_acc, predictions, targets, probs


class ModelTrainer:
    """Train and evaluate models"""

    def __init__(self, config: Config):
        self.config = config
        self.device = config.DEVICE

    def train_model(self, model, train_loader, val_loader, model_name, num_epochs=None):
        """Train a model"""
        if num_epochs is None:
            num_epochs = self.config.NUM_EPOCHS

        print(f"\n{'='*80}")
        print(f"TRAINING {model_name}")
        print(f"{'='*80}\n")

        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )

        best_acc = 0.0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 40)

            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, self.device
            )

            val_loss, val_acc, _, _, _ = validate(
                model, val_loader, criterion, self.device
            )

            scheduler.step(val_loss)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), self.config.OUTPUT_DIR / f'{model_name}_best.pth')
                print(f'✓ Model saved with accuracy: {best_acc:.4f}')

        model.load_state_dict(torch.load(self.config.OUTPUT_DIR / f'{model_name}_best.pth'))
        return model, history

    def finetune_model(self, model, train_loader, val_loader, model_name, num_epochs=15):
        """Fine-tune pretrained model"""
        print(f"\n{'='*80}")
        print(f"FINE-TUNING {model_name}")
        print(f"{'='*80}\n")

        if 'resnet' in model_name.lower():
            ct = 0
            for child in model.children():
                ct += 1
                if ct >= 8:  # Unfreeze layer4 and fc
                    for param in child.parameters():
                        param.requires_grad = True
        else:
            for param in model.parameters():
                param.requires_grad = True

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,}")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.config.LEARNING_RATE_FINETUNE
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )

        best_acc = 0.0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 40)

            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, self.device
            )
            val_loss, val_acc, _, _, _ = validate(
                model, val_loader, criterion, self.device
            )

            scheduler.step(val_loss)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), self.config.OUTPUT_DIR / f'{model_name}_finetuned.pth')
                print(f'✓ Model saved with accuracy: {best_acc:.4f}')

        model.load_state_dict(torch.load(self.config.OUTPUT_DIR / f'{model_name}_finetuned.pth'))
        return model, history

    def evaluate_model(self, model, test_loader, model_name):
        """Comprehensive evaluation"""
        print(f"\n{'='*80}")
        print(f"EVALUATING {model_name}")
        print(f"{'='*80}\n")

        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, predictions, targets, probs = validate(
            model, test_loader, criterion, self.device
        )

        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted'
        )

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1-Score: {f1:.4f}")

        print("\nClassification Report:")
        print(classification_report(targets, predictions, target_names=self.config.CLASS_NAMES))

        results = {
            'model_name': model_name,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': predictions,
            'targets': targets,
            'probs': np.array(probs)
        }
        return results


class Visualizer:
    """Visualization utilities"""

    def __init__(self, config: Config):
        self.config = config
        self.colors = ['#FF6B6B', '#16A085', '#2C3E50']

    def plot_training_history(self, histories: Dict, save_path: str):
        """Plot training history for all models"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('Training History Comparison', fontsize=16, fontweight='bold')

        for i, (model_name, history) in enumerate(histories.items()):
            axes[0].plot(history['train_loss'], label=f'{model_name} (Train)', color=self.colors[i], linewidth=2)
            axes[0].plot(history['val_loss'], label=f'{model_name} (Val)', color=self.colors[i], linewidth=2, linestyle='--')

            axes[1].plot(history['train_acc'], label=f'{model_name} (Train)', color=self.colors[i], linewidth=2)
            axes[1].plot(history['val_acc'], label=f'{model_name} (Val)', color=self.colors[i], linewidth=2, linestyle='--')

        axes[0].set_xlabel('Epoch', fontweight='bold')
        axes[0].set_ylabel('Loss', fontweight='bold')
        axes[0].set_title('Loss', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Epoch', fontweight='bold')
        axes[1].set_ylabel('Accuracy', fontweight='bold')
        axes[1].set_title('Accuracy', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved training history: {save_path}")

    def plot_confusion_matrices(self, results_list: List[Dict], save_path: str):
        """Plot confusion matrices for all models"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Confusion Matrices Comparison', fontsize=16, fontweight='bold')

        for idx, results in enumerate(results_list):
            cm = confusion_matrix(results['targets'], results['predictions'])

            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=self.config.CLASS_NAMES,
                yticklabels=self.config.CLASS_NAMES,
                ax=axes[idx], cbar_kws={'shrink': 0.8}
            )

            axes[idx].set_title(f"{results['model_name']}\nAcc: {results['test_acc']:.3f}", fontweight='bold')
            axes[idx].set_xlabel('Predicted', fontweight='bold')
            axes[idx].set_ylabel('Actual', fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved confusion matrices: {save_path}")

    def plot_roc_curves(self, results_list: List[Dict], save_path: str):
        """Plot ROC curves for all models"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('ROC Curves: Multi-class Classification', fontsize=16, fontweight='bold')

        for idx, results in enumerate(results_list):
            y_true_bin = label_binarize(results['targets'], classes=[0, 1, 2])

            for i, class_name in enumerate(self.config.CLASS_NAMES):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], results['probs'][:, i])
                roc_auc = auc(fpr, tpr)
                axes[idx].plot(fpr, tpr, color=self.colors[i], linewidth=2,
                               label=f'{class_name} (AUC = {roc_auc:.3f})')

            axes[idx].plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.3)
            axes[idx].set_xlabel('False Positive Rate', fontweight='bold')
            axes[idx].set_ylabel('True Positive Rate', fontweight='bold')
            axes[idx].set_title(results['model_name'], fontweight='bold')
            axes[idx].legend(loc='lower right')
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved ROC curves: {save_path}")

    def plot_comparison(self, results_list: List[Dict], save_path: str):
        """Plot model comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        model_names = [r['model_name'] for r in results_list]
        metrics = ['test_acc', 'precision', 'recall', 'f1']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

        x = np.arange(len(metric_labels))
        width = 0.25

        for i, results in enumerate(results_list):
            values = [results[m] for m in metrics]
            ax1.bar(x + i * width, values, width, label=results['model_name'],
                    color=self.colors[i], edgecolor='black')

        ax1.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax1.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(metric_labels)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0.85, 1.0])

        param_counts = [7.2, 25.6, 5.3]  # Millions (your placeholders)
        bars = ax2.bar(model_names, param_counts, color=self.colors,
                       edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Parameters (Millions)', fontsize=12, fontweight='bold')
        ax2.set_title('Model Complexity', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}M', ha='center', va='bottom',
                     fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved comparison: {save_path}")


def main():
    """Main training pipeline"""
    print("="*80)
    print("Medical Image Classification: Lung Disease Detection")
    print("Using PyTorch")
    print("="*80)

    config = Config()
    print(f"\nDevice: {config.DEVICE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Image Size: {config.IMG_SIZE}")
    print(f"Epochs: {config.NUM_EPOCHS}")

    dataset_dir = config.DATA_DIR / 'COVID-19_Radiography_Dataset'
    organized_dir = config.DATA_DIR / 'organized'

    # Verify dataset exists
    if not organized_dir.exists():
        print(f"\nError: Organized dataset not found at {organized_dir}")
        print("Please run organize_dataset.py first to organize the dataset.")
        sys.exit(1)

    print("\nDataset already organized. Counting samples...")
    dataset_stats = {}
    for class_name in config.CLASS_NAMES:
        class_dir_name = class_name.replace(' ', '_')
        train_count = len(list((organized_dir / 'train' / class_dir_name).glob('*.png')))
        val_count = len(list((organized_dir / 'val' / class_dir_name).glob('*.png')))
        test_count = len(list((organized_dir / 'test' / class_dir_name).glob('*.png')))
        dataset_stats[class_name] = {
            'train': train_count,
            'val': val_count,
            'test': test_count,
            'total': train_count + val_count + test_count
        }
        print(f"{class_name}: {train_count} train, {val_count} val, {test_count} test")

    print("\nCreating data loaders...")
    train_dataset = ImageFolder(organized_dir / 'train', transform=get_transforms(config, augment=True))
    val_dataset = ImageFolder(organized_dir / 'val', transform=get_transforms(config, augment=False))
    test_dataset = ImageFolder(organized_dir / 'test', transform=get_transforms(config, augment=False))

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    trainer = ModelTrainer(config)
    visualizer = Visualizer(config)

    all_histories = {}
    all_results = []
    training_times = {}

    # 1) Traditional CNN
    print("\n" + "="*80)
    print("MODEL 1: TRADITIONAL CNN")
    print("="*80)
    start_time = time.time()
    cnn_model = TraditionalCNN(num_classes=config.NUM_CLASSES)
    cnn_model, cnn_history = trainer.train_model(cnn_model, train_loader, val_loader, 'traditional_cnn')
    cnn_results = trainer.evaluate_model(cnn_model, test_loader, 'Traditional CNN')
    cnn_time = time.time() - start_time
    training_times['Traditional CNN'] = f"{cnn_time/60:.1f} minutes"
    all_histories['Traditional CNN'] = cnn_history
    all_results.append(cnn_results)

    # 2) ResNet50 (frozen + finetune)
    print("\n" + "="*80)
    print("MODEL 2: RESNET50")
    print("="*80)
    start_time = time.time()
    resnet_model = create_resnet50(num_classes=config.NUM_CLASSES)

    resnet_model, resnet_history_p1 = trainer.train_model(
        resnet_model, train_loader, val_loader, 'resnet50_frozen', num_epochs=10
    )

    resnet_model, resnet_history_p2 = trainer.finetune_model(
        resnet_model, train_loader, val_loader, 'resnet50', num_epochs=15
    )

    resnet_history = {
        'train_loss': resnet_history_p1['train_loss'] + resnet_history_p2['train_loss'],
        'train_acc': resnet_history_p1['train_acc'] + resnet_history_p2['train_acc'],
        'val_loss': resnet_history_p1['val_loss'] + resnet_history_p2['val_loss'],
        'val_acc': resnet_history_p1['val_acc'] + resnet_history_p2['val_acc']
    }

    resnet_results = trainer.evaluate_model(resnet_model, test_loader, 'ResNet50')
    resnet_time = time.time() - start_time
    training_times['ResNet50'] = f"{resnet_time/60:.1f} minutes"
    all_histories['ResNet50'] = resnet_history
    all_results.append(resnet_results)

    # 3) EfficientNet (frozen + finetune)
    print("\n" + "="*80)
    print("MODEL 3: EFFICIENTNET-B0")
    print("="*80)
    start_time = time.time()
    efficientnet_model = create_efficientnet(num_classes=config.NUM_CLASSES)

    efficientnet_model, eff_history_p1 = trainer.train_model(
        efficientnet_model, train_loader, val_loader, 'efficientnet_frozen', num_epochs=10
    )

    efficientnet_model, eff_history_p2 = trainer.finetune_model(
        efficientnet_model, train_loader, val_loader, 'efficientnet', num_epochs=15
    )

    eff_history = {
        'train_loss': eff_history_p1['train_loss'] + eff_history_p2['train_loss'],
        'train_acc': eff_history_p1['train_acc'] + eff_history_p2['train_acc'],
        'val_loss': eff_history_p1['val_loss'] + eff_history_p2['val_loss'],
        'val_acc': eff_history_p1['val_acc'] + eff_history_p2['val_acc']
    }

    eff_results = trainer.evaluate_model(efficientnet_model, test_loader, 'EfficientNetB0')
    eff_time = time.time() - start_time
    training_times['EfficientNetB0'] = f"{eff_time/60:.1f} minutes"
    all_histories['EfficientNetB0'] = eff_history
    all_results.append(eff_results)

    # Visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    visualizer.plot_training_history(all_histories, config.OUTPUT_DIR / 'training_history.png')
    visualizer.plot_confusion_matrices(all_results, config.OUTPUT_DIR / 'confusion_matrices.png')
    visualizer.plot_roc_curves(all_results, config.OUTPUT_DIR / 'roc_curves.png')
    visualizer.plot_comparison(all_results, config.OUTPUT_DIR / 'model_comparison.png')

    # Save results summary JSON
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    summary = {
        'models': {},
        'best_model': max(all_results, key=lambda x: x['test_acc'])['model_name']
    }

    for results in all_results:
        summary['models'][results['model_name']] = {
            'accuracy': float(results['test_acc']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1_score': float(results['f1'])
        }

    with open(config.OUTPUT_DIR / 'results_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"✓ Results saved to {config.OUTPUT_DIR / 'results_summary.json'}")

    # Print final summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"\nBest Model: {summary['best_model']}")
    print("\nModel Comparison:")
    print("-" * 80)
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 80)
    for model_name, metrics in summary['models'].items():
        print(f"{model_name:<20} {metrics['accuracy']:<12.4f} "
              f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
              f"{metrics['f1_score']:<12.4f}")
    print("-" * 80)

    print("\n✓ Training complete! All models + plots saved in:", config.OUTPUT_DIR)


if __name__ == '__main__':
    main()
