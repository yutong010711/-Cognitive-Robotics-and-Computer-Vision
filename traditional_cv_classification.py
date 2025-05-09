import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

# Path configuration
PROCESSED_PATH = "/home/mscrobotics2425laptop12/Desktop/vision/rgbd-dataset/processed"
TRADITIONAL_PATH = os.path.join(PROCESSED_PATH, "traditional")

# 1. Load and prepare dataset
def load_dataset(max_samples_per_class=1000, target_size=(256, 256)):
    categories = [d for d in os.listdir(TRADITIONAL_PATH) 
                  if os.path.isdir(os.path.join(TRADITIONAL_PATH, d))]
    
    # Sort categories alphabetically
    categories.sort()

    X = []
    y = []
    label_to_name = {}

    print("Loading dataset...")
    
    # Use the first 10 categories
    for label, category in enumerate(categories[:10]):
        label_to_name[label] = category
        category_path = os.path.join(TRADITIONAL_PATH, category)
        samples = 0

        img_files = os.listdir(category_path)
        
        # Limit to first 1000 samples per class
        img_files = img_files[:max_samples_per_class]

        for img_file in img_files:
            img_path = os.path.join(category_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.resize(img, target_size)
                X.append(img)
                y.append(label)
                samples += 1

        print(f"Loaded {samples} samples from {category}")

    return np.array(X), np.array(y), label_to_name


# 2. Feature extraction (SIFT + Bag of Words)
def extract_features(images, n_clusters=10000):
    print("\nExtracting SIFT features...")
    sift = cv2.SIFT_create()
    descriptors_all = []
    
    # Extract SIFT features from all images
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is not None:
            descriptors_all.append(descriptors)
    
    # Stack all descriptors for K-means clustering
    descriptors_all = np.vstack(descriptors_all)
    print(f"Total descriptors: {descriptors_all.shape}")

    # K-means clustering to create visual vocabulary
    print("Creating visual vocabulary...")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    _, labels, centers = cv2.kmeans(
        descriptors_all.astype(np.float32), 
        n_clusters, 
        None, 
        criteria, 
        10, 
        cv2.KMEANS_RANDOM_CENTERS
    )
    
    # Create BoW histogram for each image
    print("Creating BoW histograms...")
    X_bow = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, descriptors = sift.detectAndCompute(gray, None)
        hist = np.zeros(n_clusters)
        
        if descriptors is not None:
            distances = np.linalg.norm(
                descriptors[:, np.newaxis] - centers, 
                axis=2
            )
            nearest_clusters = np.argmin(distances, axis=1)
            for cluster in nearest_clusters:
                hist[cluster] += 1
            hist /= hist.sum()  # Normalize histogram
        
        X_bow.append(hist)
    
    return np.array(X_bow)

# 3. Classification and evaluation
def train_and_evaluate(X, y, label_to_name):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train SVM
    print("\nTraining SVM classifier...")
    clf = svm.SVC(kernel='linear', C=1, probability=True)
    clf.fit(X_train, y_train)
    
    # Evaluate performance
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(
        y_test, 
        y_pred, 
        target_names=list(label_to_name.values())
    ))
    
    return clf, X_test, y_test, y_pred

# 4. Visualization
def visualize_results(X, y, y_pred, label_to_name):
    plt.figure(figsize=(15, 10))
    
    # Dimensionality reduction (PCA + t-SNE)
    print("\nRunning dimensionality reduction for visualization...")
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X)
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)
    
    # Plot true labels
    plt.subplot(1, 2, 1)
    for label in np.unique(y):
        mask = y == label
        plt.scatter(
            X_tsne[mask, 0], X_tsne[mask, 1], 
            label=label_to_name[label],
            alpha=0.6
        )
    plt.title("True Labels")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot predicted labels
    plt.subplot(1, 2, 2)
    for label in np.unique(y_pred):
        mask = y_pred == label
        plt.scatter(
            X_tsne[mask, 0], X_tsne[mask, 1], 
            label=label_to_name.get(label, f"Class {label}"),
            alpha=0.6
        )
    plt.title("Predicted Labels")
    
    plt.tight_layout()
    plt.savefig("traditional_cv_results.png", bbox_inches='tight', dpi=300)
    plt.show()
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=list(label_to_name.values()),
        yticklabels=list(label_to_name.values())
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()

# Main entry
def main():
    # Load dataset (limit per class for demonstration speed)
    X, y, label_to_name = load_dataset(max_samples_per_class=1000)
    
    # Extract features
    X_features = extract_features(X, n_clusters=10000)
    
    # Train and evaluate model
    clf, X_test, y_test, y_pred = train_and_evaluate(
        X_features, y, label_to_name
    )
    
    # Visualize results
    visualize_results(X_test, y_test, y_pred, label_to_name)

if __name__ == "__main__":
    main()
