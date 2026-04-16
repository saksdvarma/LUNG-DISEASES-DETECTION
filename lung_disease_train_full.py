# -*- coding: utf-8 -*-
"""
Lung Disease Detection - Complete Training Script
Updated for TensorFlow 2.17 and Keras 3
Supports both GPU (CUDA) and CPU training
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("LUNG DISEASE DETECTION - MEDICAL IMAGE CLASSIFICATION")
print("="*80)

# Step 1: Import libraries
print("\n[Step 1/12] Importing libraries...")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import shutil

# TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[OK] GPU detected: {len(gpus)} device(s)")
        print(f"[OK] GPU will be used for training")
    except RuntimeError as e:
        print(f"Warning: {e}")
else:
    print("[INFO] No GPU detected - using CPU (training will be slower)")
    print("[INFO] To enable GPU: Install CUDA Toolkit 11.8 or 12.x from NVIDIA")

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 25
NUM_CLASSES = 3
CLASSES = ['COVID', 'Normal', 'Viral Pneumonia']

#  Step 2: Check dataset
print("\n[Step 2/12] Checking dataset...")
data_dir = Path('./data/COVID-19_Radiography_Dataset')

if not data_dir.exists():
    print("[ERROR] Dataset not found!")
    print("Please download the dataset from:")
    print("https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database")
    print(f"Extract it to: {data_dir.absolute()}")
    sys.exit(1)

# Check class directories
class_dirs = {
    'COVID': data_dir / 'COVID/images',
    'Normal': data_dir / 'Normal/images',
    'Viral Pneumonia': data_dir / 'Viral Pneumonia/images'
}

class_counts = {}
for class_name, class_path in class_dirs.items():
    if class_path.exists():
        images = list(class_path.glob('*.png'))
        class_counts[class_name] = len(images)
        print(f"  {class_name}: {len(images)} images")
    else:
        print(f"[ERROR] {class_path} not found")
        sys.exit(1)

total_images = sum(class_counts.values())
print(f"Total images: {total_images}")

# Step 3: Visualize class distribution
print("\n[Step 3/12] Creating class distribution visualization...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']

# Bar plot
ax1.bar(class_counts.keys(), class_counts.values(), color=colors, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Class', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
ax1.set_title('Dataset Distribution', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

for i, (k, v) in enumerate(class_counts.items()):
    ax1.text(i, v, str(v), ha='center', va='bottom', fontsize=11, fontweight='bold')

# Pie chart
ax2.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%',
        colors=colors, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
ax2.set_title('Class Distribution (%)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: class_distribution.png")
plt.close()

# Step 4: Organize dataset
print("\n[Step 4/12] Organizing dataset into train/val/test splits...")
organized_dir = Path('./data/organized')
train_dir = organized_dir / 'train'
val_dir = organized_dir / 'val'
test_dir = organized_dir / 'test'

# Create directories
for split_dir in [train_dir, val_dir, test_dir]:
    for class_name in CLASSES:
        (split_dir / class_name).mkdir(parents=True, exist_ok=True)

# Split and copy images (70% train, 15% val, 15% test)
for class_name, class_path in class_dirs.items():
    images = list(class_path.glob('*.png'))
    train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

    # Copy images
    for img_path in train_imgs:
        dest = train_dir / class_name / img_path.name
        if not dest.exists():
            shutil.copy(img_path, dest)

    for img_path in val_imgs:
        dest = val_dir / class_name / img_path.name
        if not dest.exists():
            shutil.copy(img_path, dest)

    for img_path in test_imgs:
        dest = test_dir / class_name / img_path.name
        if not dest.exists():
            shutil.copy(img_path, dest)

    print(f"  {class_name}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")

print("[OK] Dataset split complete")

# Step 5: Create data generators
print("\n[Step 5/12] Creating data generators with augmentation...")

# Keras 3 uses keras.utils.image_dataset_from_directory
# But for compatibility with augmentation, we'll use ImageDataGenerator from keras.preprocessing

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.9, 1.1]
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"Training samples: {train_generator.n}")
print(f"Validation samples: {val_generator.n}")
print(f"Test samples: {test_generator.n}")
print(f"Class indices: {train_generator.class_indices}")

# Step 6: Build Traditional CNN
print("\n[Step 6/12] Building Traditional CNN model...")

def create_traditional_cnn():
    """Traditional CNN architecture as baseline"""
    from keras import layers, models

    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                     input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    return model

cnn_model = create_traditional_cnn()
print(f"Total parameters: {cnn_model.count_params():,}")

# Step 7: Build ResNet50
print("\n[Step 7/12] Building ResNet50 with transfer learning...")

def create_resnet50_model():
    """ResNet50 with transfer learning"""
    from keras.applications import ResNet50
    from keras import layers, Model, Input

    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    base_model.trainable = False

    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model, base_model

resnet_model, resnet_base = create_resnet50_model()
print(f"Total parameters: {resnet_model.count_params():,}")

# Step 8: Build EfficientNetB0
print("\n[Step 8/12] Building EfficientNetB0 with transfer learning...")

def create_efficientnet_model():
    """EfficientNetB0 with transfer learning"""
    from keras.applications import EfficientNetB0
    from keras import layers, Model, Input

    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    base_model.trainable = False

    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model, base_model

efficientnet_model, efficientnet_base = create_efficientnet_model()
print(f"Total parameters: {efficientnet_model.count_params():,}")

# Step 9: Setup callbacks
print("\n[Step 9/12] Setting up training callbacks...")

def get_callbacks(model_name):
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            f'best_{model_name}.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

def compile_model(model, learning_rate=0.001):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )

print("[OK] Callbacks configured")

# Step 10: Train models
print("\n[Step 10/12] Training models...")
print("="*80)

# Train Traditional CNN
print("\n[10a] Training Traditional CNN...")
print("-"*80)
compile_model(cnn_model, learning_rate=0.001)

history_cnn = cnn_model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=get_callbacks('traditional_cnn'),
    verbose=2
)

with open('history_cnn.json', 'w') as f:
    json.dump(history_cnn.history, f)
print("[OK] Traditional CNN training complete")

# Train ResNet50 - Phase 1
print("\n[10b] Training ResNet50 - Phase 1 (Frozen base)...")
print("-"*80)
compile_model(resnet_model, learning_rate=0.001)

history_resnet_p1 = resnet_model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=get_callbacks('resnet50_phase1'),
    verbose=2
)

# Train ResNet50 - Phase 2 (Fine-tuning)
print("\n[10c] Training ResNet50 - Phase 2 (Fine-tuning)...")
print("-"*80)
resnet_base.trainable = True
for layer in resnet_base.layers[:-20]:
    layer.trainable = False

compile_model(resnet_model, learning_rate=0.0001)

history_resnet_p2 = resnet_model.fit(
    train_generator,
    epochs=EPOCHS-10,
    validation_data=val_generator,
    callbacks=get_callbacks('resnet50_finetuned'),
    verbose=2
)

# Combine histories
history_resnet = {
    key: history_resnet_p1.history[key] + history_resnet_p2.history[key]
    for key in history_resnet_p1.history.keys()
}

with open('history_resnet.json', 'w') as f:
    json.dump(history_resnet, f)
print("[OK] ResNet50 training complete")

# Train EfficientNet - Phase 1
print("\n[10d] Training EfficientNetB0 - Phase 1 (Frozen base)...")
print("-"*80)
compile_model(efficientnet_model, learning_rate=0.001)

history_efficientnet_p1 = efficientnet_model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=get_callbacks('efficientnet_phase1'),
    verbose=2
)

# Train EfficientNet - Phase 2 (Fine-tuning)
print("\n[10e] Training EfficientNetB0 - Phase 2 (Fine-tuning)...")
print("-"*80)
efficientnet_base.trainable = True
for layer in efficientnet_base.layers[:-20]:
    layer.trainable = False

compile_model(efficientnet_model, learning_rate=0.0001)

history_efficientnet_p2 = efficientnet_model.fit(
    train_generator,
    epochs=EPOCHS-10,
    validation_data=val_generator,
    callbacks=get_callbacks('efficientnet_finetuned'),
    verbose=2
)

# Combine histories
history_efficientnet = {
    key: history_efficientnet_p1.history[key] + history_efficientnet_p2.history[key]
    for key in history_efficientnet_p1.history.keys()
}

with open('history_efficientnet.json', 'w') as f:
    json.dump(history_efficientnet, f)
print("[OK] EfficientNetB0 training complete")

# Step 11: Evaluate models
print("\n[Step 11/12] Evaluating models on test set...")
print("="*80)

def evaluate_model(model, model_name, test_gen):
    """Comprehensive model evaluation"""
    print(f"\nEvaluating {model_name}...")
    print("-"*80)

    test_gen.reset()
    y_pred_prob = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = test_gen.classes

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    print(f"Test Accuracy:  {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall:    {recall:.4f}")
    print(f"Test F1-Score:  {f1:.4f}")

    class_names = list(test_gen.class_indices.keys())
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_names': class_names
    }

results_cnn = evaluate_model(cnn_model, "Traditional CNN", test_generator)
results_resnet = evaluate_model(resnet_model, "ResNet50", test_generator)
results_efficientnet = evaluate_model(efficientnet_model, "EfficientNetB0", test_generator)

# Step 12: Create visualizations
print("\n[Step 12/12] Creating visualizations...")

# Training history comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Training History Comparison', fontsize=18, fontweight='bold')

models_history = {
    'Traditional CNN': history_cnn.history,
    'ResNet50': history_resnet,
    'EfficientNetB0': history_efficientnet
}

colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']

# Accuracy
for idx, (model_name, history) in enumerate(models_history.items()):
    axes[0, 0].plot(history['accuracy'], label=f'{model_name} (Train)',
                    color=colors[idx], linewidth=2)
    axes[0, 0].plot(history['val_accuracy'], label=f'{model_name} (Val)',
                    color=colors[idx], linewidth=2, linestyle='--')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Accuracy')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# Loss
for idx, (model_name, history) in enumerate(models_history.items()):
    axes[0, 1].plot(history['loss'], label=f'{model_name} (Train)',
                    color=colors[idx], linewidth=2)
    axes[0, 1].plot(history['val_loss'], label=f'{model_name} (Val)',
                    color=colors[idx], linewidth=2, linestyle='--')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('Loss')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# Precision
for idx, (model_name, history) in enumerate(models_history.items()):
    axes[1, 0].plot(history['precision'], label=f'{model_name} (Train)',
                    color=colors[idx], linewidth=2)
    axes[1, 0].plot(history['val_precision'], label=f'{model_name} (Val)',
                    color=colors[idx], linewidth=2, linestyle='--')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].set_title('Precision')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True, alpha=0.3)

# Recall
for idx, (model_name, history) in enumerate(models_history.items()):
    axes[1, 1].plot(history['recall'], label=f'{model_name} (Train)',
                    color=colors[idx], linewidth=2)
    axes[1, 1].plot(history['val_recall'], label=f'{model_name} (Val)',
                    color=colors[idx], linewidth=2, linestyle='--')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].set_title('Recall')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history_comparison.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: training_history_comparison.png")
plt.close()

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')

results_list = [
    (results_cnn, 'Traditional CNN'),
    (results_resnet, 'ResNet50'),
    (results_efficientnet, 'EfficientNetB0')
]

for idx, (results, model_name) in enumerate(results_list):
    cm = confusion_matrix(results['y_true'], results['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                xticklabels=results['class_names'],
                yticklabels=results['class_names'],
                ax=axes[idx], cbar_kws={'shrink': 0.8})
    axes[idx].set_title(f'{model_name}\nAcc: {results["accuracy"]:.3f}')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: confusion_matrices.png")
plt.close()

# Model comparison table
comparison_data = {
    'Model': ['Traditional CNN', 'ResNet50', 'EfficientNetB0'],
    'Parameters': [
        f"{cnn_model.count_params():,}",
        f"{resnet_model.count_params():,}",
        f"{efficientnet_model.count_params():,}"
    ],
    'Accuracy': [f"{results_cnn['accuracy']:.4f}", f"{results_resnet['accuracy']:.4f}", f"{results_efficientnet['accuracy']:.4f}"],
    'Precision': [f"{results_cnn['precision']:.4f}", f"{results_resnet['precision']:.4f}", f"{results_efficientnet['precision']:.4f}"],
    'Recall': [f"{results_cnn['recall']:.4f}", f"{results_resnet['recall']:.4f}", f"{results_efficientnet['recall']:.4f}"],
    'F1-Score': [f"{results_cnn['f1']:.4f}", f"{results_resnet['f1']:.4f}", f"{results_efficientnet['f1']:.4f}"]
}

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv('model_comparison.csv', index=False)
print("\n" + "="*100)
print("MODEL COMPARISON SUMMARY")
print("="*100)
print(comparison_df.to_string(index=False))
print("="*100)

# Save models
print("\n[Final] Saving models...")
cnn_model.save('traditional_cnn_final.keras')
resnet_model.save('resnet50_final.keras')
efficientnet_model.save('efficientnet_final.keras')

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  - class_distribution.png")
print("  - training_history_comparison.png")
print("  - confusion_matrices.png")
print("  - model_comparison.csv")
print("  - traditional_cnn_final.keras")
print("  - resnet50_final.keras")
print("  - efficientnet_final.keras")
print("  - history_cnn.json")
print("  - history_resnet.json")
print("  - history_efficientnet.json")
print("\n" + "="*80)
