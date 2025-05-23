import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
from itertools import product

# Constants
PROCESSED_ROOT = "/home/mscrobotics2425laptop12/Desktop/vision/rgbd-dataset/processed"
TARGET_SIZE = (256, 256)  # Reduce resolution to save memory

# Hyperparameter combinations
BATCH_SIZES = [64, 32]
FILTER_SETS = [
    [32, 64, 128],
    [64, 128, 256]
]
OPTIMIZERS = ['adam', 'sgd']
LRS = [1e-2, 1e-3]
DROPOUTS = [0.3, 0.7]

# 1. Data Generator Class
class RGBDDataGenerator(utils.Sequence):
    def __init__(self, file_paths, labels, batch_size=64, target_size=(256, 256), shuffle=True):
        """
        :param file_paths: List of .npy file paths for all samples
        :param labels: Corresponding label list
        """
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.file_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_images = []
        batch_labels = []

        for i in batch_indices:
            img = np.load(self.file_paths[i])
            if img.shape[:2] != self.target_size:
                img = tf.image.resize(img, self.target_size).numpy()
            batch_images.append(img)
            batch_labels.append(self.labels[i])

        return np.array(batch_images), np.array(batch_labels)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# 2. Build CNN model (parameterized with hyperparameters)
def build_cnn_model(input_shape=(256, 256, 3), num_classes=10, filters=[32, 64, 128], dropout_rate=0.5):
    model = models.Sequential([
        layers.Conv2D(filters[0], (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(filters[1], (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(filters[2], (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# 3. Visualization tool (displays hyperparameter results table)
def plot_results_table(results_df):
    plt.figure(figsize=(16, 8))
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False)
    
    # Create and display table
    table = pd.plotting.table(ax, results_df, loc='center', cellLoc='center', colWidths=[0.1]*len(results_df.columns))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    
    plt.title('Hyperparameter Tuning Results', pad=20)
    plt.savefig('hyperparameter_results.png', bbox_inches='tight', dpi=300)
    plt.show()

# 4. Main function (for hyperparameter testing)
def main():
    # Configure GPU (optional)
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    # Load class list (first 10 categories)
    cnn_path = os.path.join(PROCESSED_ROOT, "cnn")
    categories = sorted([d for d in os.listdir(cnn_path) 
                         if os.path.isdir(os.path.join(cnn_path, d))])[:10]
    
    # Collect all sample paths and labels
    all_paths = []
    all_labels = []

    for label, category in enumerate(categories):
        category_path = os.path.join(cnn_path, category)
        npy_files = sorted([f for f in os.listdir(category_path) if f.endswith('.npy')])[:1000]
        file_paths = [os.path.join(category_path, f) for f in npy_files]
        all_paths.extend(file_paths)
        all_labels.extend([label] * len(file_paths))
        print(f"{category}: {len(file_paths)} samples")

    # Split into training and test sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
    
    # Store all results
    results = []
    
    # Iterate through all hyperparameter combinations
    param_combinations = list(product(BATCH_SIZES, FILTER_SETS, OPTIMIZERS, LRS, DROPOUTS))
    total_combinations = len(param_combinations)
    
    print(f"\nStarting hyperparameter tuning with {total_combinations} combinations...")
    
    for i, (batch_size, filters, optimizer, lr, dropout) in enumerate(param_combinations, 1):
        print(f"\n=== Testing combination {i}/{total_combinations} ===")
        print(f"Batch: {batch_size}, Filters: {filters}, Optimizer: {optimizer}, LR: {lr}, Dropout: {dropout}")
        
        # Create data generators
        train_gen = RGBDDataGenerator(X_train, y_train, batch_size=batch_size, target_size=TARGET_SIZE, shuffle=True)
        test_gen = RGBDDataGenerator(X_test, y_test, batch_size=batch_size, target_size=TARGET_SIZE, shuffle=False)
        
        # Build model
        model = build_cnn_model(input_shape=(*TARGET_SIZE, 3), num_classes=len(categories), 
                               filters=filters, dropout_rate=dropout)
        
        # Configure optimizer
        if optimizer == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            opt = tf.keras.optimizers.SGD(learning_rate=lr)
            
        model.compile(optimizer=opt,
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        # Train model (reduced epochs for faster testing)
        history = model.fit(
            train_gen,
            validation_data=test_gen,
            epochs=50,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ]
        )
        
        # Evaluate model
        val_loss, val_acc = model.evaluate(test_gen, verbose=0)
        
        # Record results
        results.append({
            'Batch Size': batch_size,
            'Filters': '-'.join(map(str, filters)),
            'Optimizer': optimizer,
            'Learning Rate': lr,
            'Dropout': dropout,
            'Val Accuracy': val_acc,
            'Epochs Trained': len(history.history['val_accuracy'])
        })
        
        print(f"Validation Accuracy: {val_acc:.4f}")
    
    # Convert results to DataFrame and sort
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Val Accuracy', ascending=False)
    
    # Save results
    results_df.to_csv('hyperparameter_results.csv', index=False)
    
    # Visualize results
    plot_results_table(results_df)
    
    # Print best combination
    best_result = results_df.iloc[0]
    print("\n⭐ Best Hyperparameter Combination:")
    print(f"Batch Size: {best_result['Batch Size']}")
    print(f"Filters: {best_result['Filters']}")
    print(f"Optimizer: {best_result['Optimizer']}")
    print(f"Learning Rate: {best_result['Learning Rate']}")
    print(f"Dropout: {best_result['Dropout']}")
    print(f"Validation Accuracy: {best_result['Val Accuracy']:.4f}")

if __name__ == "__main__":
    main()
