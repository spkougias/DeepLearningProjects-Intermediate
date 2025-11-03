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


# -------------------------
#       Load dataset       
# -------------------------

def load_cifar10_batch(file_path):
    
    with open(file_path, 'rb') as f:
        temp_data = pickle.load(f, encoding='bytes');
        images = temp_data[b'data'];  # uint8 (N, 3072)
        labels = temp_data[b'labels']; # list length N
    return images, labels;

#Load data_batch_1..5 as train, test_batch as test
def load_cifar10(data_dir):
    
    img_train_list = [];
    label_train_list = [];
    
    for i in range(1, 6):
        batch_path = os.path.join(data_dir, f'data_batch_{i}');
        images, labels = load_cifar10_batch(batch_path);
        img_train_list.append(images);
        label_train_list.extend(labels);
        
    img_train = np.concatenate(img_train_list, axis=0);  # (50000, 3072)
    label_train = np.array(label_train_list, dtype = np.int64);

    img_test, label_test = load_cifar10_batch(os.path.join(data_dir, 'test_batch'));
    img_test = np.array(img_test, dtype = np.uint8);
    label_test = np.array(label_test, dtype = np.int64);

    return img_train, label_train, img_test, label_test;


# -------------------------
#        Pre Process       
# -------------------------

def preprocess(x, scale = True, subtract_mean = True):
    # Input X: numpy array shape (N, 3072), dtype uint8
    x = x.astype(np.float32);
    if scale:
        x /= 255.0;
    if subtract_mean:
        x -= x.mean(axis=0, keepdims = True);
    return x;


# -------------------------
#      k-NN classifier     
# -------------------------

#Compute distances
# Use chunks of the data set for faster results
# Use (x-y)^2= x^2 + y^2 -2*x*y  for distance
def compute_distances_chunked(img_test, img_train, chunk_size=200):
     # Convert to torch tensors to make the process faster
    test_t = torch.from_numpy(img_test);
    train_t = torch.from_numpy(img_train);
    n_test = test_t.shape[0];
    results = [];
    
    train_sq = (train_t*train_t).sum(dim = 1);  # compute train square
    
    for start in range(0, n_test, chunk_size):
        end = min(n_test, start+chunk_size); #end of chunk
        test_chunk = test_t[start:end];
        
        # compute: test^2 + train^2 -2*test*train^T 
        chunk_sq = (test_chunk*test_chunk).sum(dim=1).unsqueeze(1);#transform to (B,1)
        # compute: -2*(test @ train^T)
        cross = -2.0*(test_chunk @ train_t.T);
        # compute distance (x-y)^2= x^2 + y^2 -2*x*y 
        dist = chunk_sq + train_sq.unsqueeze(0) + cross;
        
        results.append((start, end, dist.numpy()));
    return results;

#Prediction k-NN, computed chunked
def knn_predict(img_train, label_train, img_test, k=1, chunk_size=200):
    
    results = compute_distances_chunked(img_test, img_train, chunk_size=chunk_size)
    n_test = img_test.shape[0];
    y_pred = np.empty(n_test, dtype=np.int64); # Hold predictions
    
    for start, end, dists in results:

        if k == 1:
            idx = np.argmin(dists, axis=1);
            y_pred[start:end] = label_train[idx];
        else:
            # get k smallest distances, this gets k smallest distance positions in the dists array 
            idx_k = np.argpartition(dists, kth=k-1, axis=1)[:, :k]; 
            # get the labels based on the above positions
            labels_k = label_train[idx_k];
            
            # majority vote
            for i in range(labels_k.shape[0]):
                label_list = labels_k[i].tolist();
                c = Counter(label_list);
                # choose most common
                most_common = c.most_common();
                top_count = most_common[0][1];
                tied = [lab for lab, cnt in most_common if cnt == top_count];
                if len(tied) == 1:
                    y_pred[start + i] = most_common[0][0];
                else:
                    # break tie by picking the label of nearest candidate
                    # find their distances
                    row = dists[i];
                    idxs = idx_k[i];
                    closest_idx = idxs[np.argmin(row[idxs])];
                    y_pred[start + i] = label_train[closest_idx];
    return y_pred;


# -------------------------
# Nearest centroid classifier
# -------------------------

def nearest_centroid_predict(img_train, label_train, img_test):
    # compute centroids
    classes = np.unique(label_train);
    n_feature = img_train.shape[1];
    centroids = np.zeros((len(classes), n_feature), dtype=np.float32);
    
    for i, c in enumerate(classes):
        centroids[i] = img_train[label_train == c].mean(axis=0);
    
    test_t = torch.from_numpy(img_test);
    centroid_t = torch.from_numpy(centroids);
    
    # compute: x^2 + c^2 - 2*x*c^T same as in k-NN
    test_sq = (test_t*test_t).sum(dim=1).unsqueeze(1);
    centroid_sq = (centroid_t*centroid_t).sum(dim=1).unsqueeze(0);
    cross = -2.0*(test_t @ centroid_t.T);
    dists = test_sq + centroid_sq + cross;
    
    pred_idx = torch.argmin(dists, dim=1).numpy();
    return classes[pred_idx];


# -------------------------
#           Main                     
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
    img_train, label_train, img_test, y_test = load_cifar10(args.data_dir);
    print("Original shapes:", img_train.shape, label_train.shape, img_test.shape, y_test.shape);

    # Optionally subsample for speed (you can set train_samples=50000/test_samples=10000 to use all)
    if args.train_samples < img_train.shape[0]:
        idx = np.random.choice(img_train.shape[0], args.train_samples, replace=False);
        img_train = img_train[idx];
        label_train = label_train[idx];
    if args.test_samples < img_test.shape[0]:
        idx = np.random.choice(img_test.shape[0], args.test_samples, replace=False);
        img_test = img_test[idx];
        y_test = y_test[idx];

    # Preprocess (float, scale to [0,1], zero-mean)
    img_train = preprocess(img_train, scale=True, subtract_mean=True);
    img_test = preprocess(img_test, scale=True, subtract_mean=True);

    # Optionally PCA
    if args.pca:
        print(f"Applying PCA -> {args.pca_dim} dims (this may take a bit)...");
        pca = PCA(n_components=args.pca_dim, svd_solver='randomized', whiten=False);
        img_train = pca.fit_transform(img_train);
        img_test = pca.transform(img_test);
        print("After PCA shapes:", img_train.shape, img_test.shape);
    else:
        print("No PCA. Using raw pixel vectors (shape per sample = {})".format(img_train.shape[1]));

    # Nearest centroid
    print("Computing nearest-centroid predictions...");
    y_centroid = nearest_centroid_predict(img_train, label_train, img_test);
    acc_centroid = accuracy_score(y_test, y_centroid);
    print(f"Nearest-centroid accuracy: {acc_centroid*100:.2f}%");

    # 1-NN
    print("Computing 1-NN predictions (chunk_size={})...".format(args.chunk_size));
    y_knn1 = knn_predict(img_train, label_train, img_test, k=1, chunk_size=args.chunk_size);
    acc_knn1 = accuracy_score(y_test, y_knn1);
    print(f"1-NN accuracy: {acc_knn1*100:.2f}%");

    # 3-NN
    print("Computing 3-NN predictions...");
    y_knn3 = knn_predict(img_train, label_train, img_test, k=3, chunk_size=args.chunk_size);
    acc_knn3 = accuracy_score(y_test, y_knn3);
    print(f"3-NN accuracy: {acc_knn3*100:.2f}%");

    # Print a small summary
    print("\nSummary (on {} train, {} test):".format(img_train.shape[0], img_test.shape[0]));
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
