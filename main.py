#Run: python main.py --data_dir ./cifar-10-batches-py

import os
import pickle
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm

#Load dataset
def load_cifar10_batch(file_path):
    
    with open(file_path, 'rb') as f:
        temp_data = pickle.load(f, encoding='bytes');
        images = temp_data[b'data'];  # uint8 (N, 3072)
        labels = temp_data[b'labels']; # list length N
    return images, labels;

#Load data_batch_1..5 as train, test_batch as test
def load_cifar10(data_dir):
    
    image_train_list = [];
    label_train_list = [];
    
    for i in range(1, 6):
        batch_path = os.path.join(data_dir, f'data_batch_{i}');
        images, labels = load_cifar10_batch(batch_path);
        image_train_list.append(images);
        label_train_list.extend(labels);
        
    image_train = np.concatenate(image_train_list, axis=0);  # (50000, 3072)
    label_train = np.array(label_train_list, dtype = np.int64);

    image_test, label_test = load_cifar10_batch(os.path.join(data_dir, 'test_batch'));
    image_test = np.array(image_test, dtype = np.uint8);
    label_test = np.array(label_test, dtype = np.int64);

    return image_train, label_train, image_test, label_test;


def preprocess(x, scale = True, subtract_mean = True):
    # Input X: numpy array shape (N, 3072), dtype uint8
    x = x.astype(np.float32);
    if scale:
        x /= 255.0;
    if subtract_mean:
        x -= x.mean(axis=0, keepdims = True);
    return x;


# k-NN
# Use chunks of the data set for faster results
# Use (x-y)^2= x^2 + y^2 -2*x*y  for distance
def compute_distances_chunked(image_test, image_train, chunk_size=200):
     # Convert to torch tensors to make the process faster
    test_t = torch.from_numpy(image_test);
    train_t = torch.from_numpy(image_train);
    N_test = test_t.shape[0];
    results = [];
    
    train_sq = (train_t*train_t).sum(dim = 1);  # compute train square
    for start in range(0, N_test, chunk_size):
        end = min(N_test, start+chunk_size); #end of chunk
        test_chunk = test_t[start:end];
        
        # compute: test^2 + train^2 -2*test*train^T 
        chunk_sq = (test_chunk*test_chunk).sum(dim=1).unsqueeze(1);#transform to (B,1)
        # compute: -2*(test @ train^T)
        cross = -2.0*(test_chunk @ train_t.T);
        # compute distance (x-y)^2= x^2 + y^2 -2*x*y 
        dist = chunk_sq + train_sq.unsqueeze(0) + cross;
        
        results.append((start, end, dist.numpy()));
    return results;

def knn_predict(image_train, label_train, image_test, k=1, chunk_size=200):
    """
    k-NN prediction using squared Euclidean distances, computed chunked.
    Returns predicted labels array of length N_test.
    """
    results = compute_distances_chunked(image_test, image_train, chunk_size=chunk_size)
    N_test = image_test.shape[0];
    y_pred = np.empty(N_test, dtype=np.int64);
    for start, end, dists in results:
        # dists shape (B, N_train)
        if k == 1:
            idx = np.argmin(dists, axis=1);
            y_pred[start:end] = label_train[idx];
        else:
            # get k smallest distances: use argpartition then sort small subset
            idx_k = np.argpartition(dists, kth=k-1, axis=1)[:, :k];  # (B, k) unordered
            # for safety, sort them by distance (not strictly necessary for voting)
            # but we only need labels here:
            labels_k = label_train[idx_k];  # (B, k)
            # majority vote (break ties by smallest distance order)
            for i in range(labels_k.shape[0]):
                lab_list = labels_k[i].tolist();
                c = Counter(lab_list);
                # choose most common; if tie, pick the label of the closest neighbor
                most_common = c.most_common();
                top_count = most_common[0][1];
                tied = [lab for lab, cnt in most_common if cnt == top_count];
                if len(tied) == 1:
                    y_pred[start + i] = most_common[0][0];
                else:
                    # break tie: pick label of nearest among the k candidates
                    # find their distances
                    row = dists[i];  # distances to all train
                    # among idx_k[i], pick the one with smallest distance
                    idxs = idx_k[i];
                    closest_idx = idxs[np.argmin(row[idxs])];
                    y_pred[start + i] = label_train[closest_idx];
    return y_pred;

# -------------------------
# Nearest centroid classifier
# -------------------------
def nearest_centroid_predict(image_train, label_train, image_test):
    # compute centroids
    classes = np.unique(label_train);
    D = image_train.shape[1];
    centroids = np.zeros((len(classes), D), dtype=np.float32);
    for i, c in enumerate(classes):
        centroids[i] = image_train[label_train == c].mean(axis=0);
    # distances: use efficient formula with torch or numpy
    Xt = torch.from_numpy(image_test);  # T x D
    C = torch.from_numpy(centroids);  # num_classes x D
    # compute: ||x||^2 + ||c||^2 - 2 x c^T
    x_sq = (Xt * Xt).sum(dim=1).unsqueeze(1); # T x 1
    c_sq = (C * C).sum(dim=1).unsqueeze(0);    # 1 x C
    cross = -2.0 * (Xt @ C.T);                 # T x C
    dists = x_sq + c_sq + cross;              # T x C
    pred_idx = torch.argmin(dists, dim=1).numpy();
    return classes[pred_idx];  # map indices to class labels

# -------------------------
# Main / CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser();
    parser.add_argument('--data_dir', type=str, required=True, help='Path to cifar-10-batches-py folder');
    parser.add_argument('--train_samples', type=int, default=5000, help='How many training samples to use (for speed). Max 50000');
    parser.add_argument('--test_samples', type=int, default=1000, help='How many test samples to use (for speed). Max 10000');
    parser.add_argument('--pca', action='store_true', help='Apply PCA to reduce dimensionality (faster kNN)');
    parser.add_argument('--pca_dim', type=int, default=200, help='PCA target dimension if --pca');
    parser.add_argument('--chunk_size', type=int, default=200, help='Chunk size for distance computation');
    args = parser.parse_args();

    print("Loading CIFAR-10 from", args.data_dir);
    image_train, label_train, image_test, y_test = load_cifar10(args.data_dir);
    print("Original shapes:", image_train.shape, label_train.shape, image_test.shape, y_test.shape);

    # Optionally subsample for speed (you can set train_samples=50000/test_samples=10000 to use all)
    if args.train_samples < image_train.shape[0]:
        idx = np.random.choice(image_train.shape[0], args.train_samples, replace=False);
        image_train = image_train[idx];
        label_train = label_train[idx];
    if args.test_samples < image_test.shape[0]:
        idx = np.random.choice(image_test.shape[0], args.test_samples, replace=False);
        image_test = image_test[idx];
        y_test = y_test[idx];

    # Preprocess (float, scale to [0,1], zero-mean)
    image_train = preprocess(image_train, scale=True, subtract_mean=True);
    image_test = preprocess(image_test, scale=True, subtract_mean=True);

    # Optionally PCA
    if args.pca:
        print(f"Applying PCA -> {args.pca_dim} dims (this may take a bit)...");
        pca = PCA(n_components=args.pca_dim, svd_solver='randomized', whiten=False);
        image_train = pca.fit_transform(image_train);
        image_test = pca.transform(image_test);
        print("After PCA shapes:", image_train.shape, image_test.shape);
    else:
        print("No PCA. Using raw pixel vectors (shape per sample = {})".format(image_train.shape[1]));

    # Nearest centroid
    print("Computing nearest-centroid predictions...");
    y_centroid = nearest_centroid_predict(image_train, label_train, image_test);
    acc_centroid = accuracy_score(y_test, y_centroid);
    print(f"Nearest-centroid accuracy: {acc_centroid*100:.2f}%");

    # 1-NN
    print("Computing 1-NN predictions (chunk_size={})...".format(args.chunk_size));
    y_knn1 = knn_predict(image_train, label_train, image_test, k=1, chunk_size=args.chunk_size);
    acc_knn1 = accuracy_score(y_test, y_knn1);
    print(f"1-NN accuracy: {acc_knn1*100:.2f}%");

    # 3-NN
    print("Computing 3-NN predictions...");
    y_knn3 = knn_predict(image_train, label_train, image_test, k=3, chunk_size=args.chunk_size);
    acc_knn3 = accuracy_score(y_test, y_knn3);
    print(f"3-NN accuracy: {acc_knn3*100:.2f}%");

    # Print a small summary
    print("\nSummary (on {} train, {} test):".format(image_train.shape[0], image_test.shape[0]));
    print(f"Nearest-centroid: {acc_centroid*100:.2f}%");
    print(f"1-NN:             {acc_knn1*100:.2f}%");
    print(f"3-NN:             {acc_knn3*100:.2f}%");

    # Confusion matrix for best classifier
    best = '1-NN' if acc_knn1 >= max(acc_knn3, acc_centroid) else ('3-NN' if acc_knn3 >= max(acc_knn1, acc_centroid) else 'centroid');
    print("Best classifier:", best);

    if best == '1-NN':
        y_best = y_knn1;
    elif best == '3-NN':
        y_best = y_knn3;
    else:
        y_best = y_centroid;

    cm = confusion_matrix(y_test, y_best);
    print("Confusion matrix (rows=true, cols=predicted):");
    print(cm);

    # Optionally show a heatmap
    """
    try:
        plt.figure(figsize=(8, 8));
        plt.imshow(cm, interpolation='nearest');
        plt.title(f'Confusion matrix ({best})');
        plt.colorbar();
        plt.xlabel('Predicted');
        plt.ylabel('True');
        plt.tight_layout();
        plt.show();
    except Exception:
        pass;
    """

if __name__ == '__main__':
    main();
