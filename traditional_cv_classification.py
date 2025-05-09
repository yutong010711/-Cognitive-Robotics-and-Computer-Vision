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

# 常量定义
PROCESSED_ROOT = "/home/mscrobotics2425laptop12/Desktop/vision/rgbd-dataset/processed"
BATCH_SIZE = 32
TARGET_SIZE = (256, 256)  # 降低分辨率节省内存
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.7

# 1. 数据生成器类
class RGBDDataGenerator(utils.Sequence):
    def __init__(self, file_paths, labels, batch_size=32, target_size=(256, 256), shuffle=True):
        """
        :param file_paths: 所有样本的 .npy 文件路径列表
        :param labels: 对应的标签列表
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

# 2. 构建CNN模型
def build_cnn_model(input_shape=(256, 256, 3), num_classes=10):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(DROPOUT_RATE),  # 使用指定的dropout率
        layers.Dense(num_classes, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)  # 使用指定的学习率
    model.compile(optimizer=optimizer,
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# 3. 可视化工具
def plot_results(history, y_true, y_pred, class_names):
    plt.figure(figsize=(24, 6))
    
    # 训练曲线 - Accuracy
    plt.subplot(1, 4, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 训练曲线 - Loss
    plt.subplot(1, 4, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 混淆矩阵
    plt.subplot(1, 4, 3)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', 
               xticklabels=class_names,
               yticklabels=class_names,
               cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    
    # 分类报告
    plt.subplot(1, 4, 4)
    report = classification_report(y_true, y_pred, 
                                 target_names=class_names,
                                 output_dict=True)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, 
               annot=True, cmap='YlGnBu')
    plt.title('Classification Report')
    
    plt.tight_layout()
    plt.savefig('cnn_results.png', bbox_inches='tight', dpi=300)
    plt.show()

# 4. 主流程
def main():
    # 配置GPU（可选）
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    # 加载类别列表（前10类）
    cnn_path = os.path.join(PROCESSED_ROOT, "cnn")
    categories = sorted([d for d in os.listdir(cnn_path) 
                         if os.path.isdir(os.path.join(cnn_path, d))])[:10]
    
    # 收集所有样本路径和标签
    all_paths = []
    all_labels = []

    for label, category in enumerate(categories):
        category_path = os.path.join(cnn_path, category)
        npy_files = sorted([f for f in os.listdir(category_path) if f.endswith('.npy')])[:1000]
        file_paths = [os.path.join(category_path, f) for f in npy_files]
        all_paths.extend(file_paths)
        all_labels.extend([label] * len(file_paths))
        print(f"{category}: {len(file_paths)} samples")

    # 划分训练集和测试集（80/20）
    X_train, X_test, y_train, y_test = train_test_split(
        all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels)

    # 创建数据生成器
    train_gen = RGBDDataGenerator(X_train, y_train, batch_size=BATCH_SIZE, target_size=TARGET_SIZE, shuffle=True)
    test_gen = RGBDDataGenerator(X_test, y_test, batch_size=BATCH_SIZE, target_size=TARGET_SIZE, shuffle=False)

    # 构建模型
    model = build_cnn_model(input_shape=(*TARGET_SIZE, 3), num_classes=len(categories))
    model.summary()

    # 训练模型
    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=50,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
        ]
    )

    # 测试并可视化
    y_pred = model.predict(test_gen).argmax(axis=1)
    y_true = np.array(y_test[:len(y_pred)])

    print("\n✅ Test Accuracy:", np.mean(y_pred == y_true))
    plot_results(history, y_true, y_pred, categories)


if __name__ == "__main__":
    main()