import torch
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from utils import get_cifar10_loaders
from models import initialize_model
from train_evaluate import train_model

def plot_experiment_results(all_histories, model_name):

    # Convert list of dicts to dict of lists
    keys = ['train_loss', 'val_loss', 'test_loss', 'test_acc']
    agg_data = {k: [] for k in keys}
    
    for history in all_histories:
        for k in keys:
            agg_data[k].append(history[k])
            
    # Convert to Numpy for easy math
    for k in keys:
        agg_data[k] = np.array(agg_data[k])

    epochs_range = range(1, agg_data['train_loss'].shape[1] + 1)
    
    plt.figure(figsize=(14, 6))
    
    # --- Plot 1: Loss (Mean +/- Std) ---
    plt.subplot(1, 2, 1)
    
    # Calculate Stats
    train_mean = np.mean(agg_data['train_loss'], axis=0)
    train_std = np.std(agg_data['train_loss'], axis=0)
    val_mean = np.mean(agg_data['val_loss'], axis=0)
    val_std = np.std(agg_data['val_loss'], axis=0)
    test_loss_mean = np.mean(agg_data['test_loss'], axis=0)
    test_loss_std = np.std(agg_data['test_loss'], axis=0)
    
    # Plot Training
    plt.plot(epochs_range, train_mean, label='Train Loss (Mean)', color='blue')
    plt.fill_between(epochs_range, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.2)
    
    # Plot Validation
    plt.plot(epochs_range, val_mean, label='Val Loss (Mean)', color='green', linestyle='--')
    plt.fill_between(epochs_range, val_mean - val_std, val_mean + val_std, color='green', alpha=0.2)

    # Plot Testing (New)
    plt.plot(epochs_range, test_loss_mean, label='Test Loss (Mean)', color='red', linestyle=':')
    plt.fill_between(epochs_range, test_loss_mean - test_loss_std, test_loss_mean + test_loss_std, color='red', alpha=0.2)

    plt.title(f'{model_name} Stability Analysis (Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # --- Plot 2: Test Accuracy (Mean +/- Std) ---
    plt.subplot(1, 2, 2)
    
    test_acc_mean = np.mean(agg_data['test_acc'], axis=0)
    test_acc_std = np.std(agg_data['test_acc'], axis=0)
    
    plt.plot(epochs_range, test_acc_mean, label='Test Acc (Mean)', color='red')
    plt.fill_between(epochs_range, test_acc_mean - test_acc_std, test_acc_mean + test_acc_std, color='red', alpha=0.2)
    
    plt.title(f'{model_name} Stability Analysis (Accuracy)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    filename = f'{model_name}_experiment_analysis.png'
    plt.tight_layout()
    plt.savefig(filename)
    print(f"\nAnalysis plot saved as {filename}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Deep Learning Experiment Runner - Variance Analysis")
    
    parser.add_argument('--model', type=str, default='simple_cnn', 
                        choices=['simple_mlp', 'complex_mlp', 'simple_cnn', 'complex_cnn', 'transfer'], 
                        help='Choose the architecture to train')
    parser.add_argument('--criterion', type=str, default='cross_entropy', 
                        choices=['cross_entropy', 'label_smoothing', 'hinge_loss'], 
                        help='Choose the criterion used in training')
    parser.add_argument('--optimizer', type=str, default='adam', 
                        choices=['adam', 'adamw', 'sgd', 'sgd_vanilla', 'rmsprop'], 
                        help='Choose the optimizer used in training')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'leaky_relu', 'sigmoid'],
                        help='Choose the activation function')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--runs', type=int, default=3, help='Number of times to repeat the experiment')
    
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"SUCCESS: GPU found: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("WARNING: GPU NOT found. Using CPU.")
    print("------------------------------------------------")
    print("\nLoading Data...")

    train_loader, val_loader, test_loader, classes = get_cifar10_loaders(
        batch_size=args.batch_size, augment=args.augment
    )

    all_histories = []


    print("\nLoading Data...\n")
    print("##########################################################################################################")
    print(f"> Settings: Epochs: {args.epochs} | Learning rate: {args.lr} | Batch Size: {args.batch_size} | Dropout: {args.dropout} | Augment: {args.augment}")
    print(f"> Optimizer: {args.optimizer}")
    print(f"> Criterion: {args.criterion}")
    print(f"> Activation: {args.activation}")
    print(f"> Running {args.runs} times...")
    print("##########################################################################################################\n")
    
    for run in range(args.runs):
        print(f"\n--- Run {run + 1} / {args.runs} ---")
        
        # Initialize using the helper from models.py
        model = initialize_model(args.model, args.dropout, device)

        # Train
        history = train_model(
            model, train_loader, val_loader, test_loader, device, 
            epochs=args.epochs, lr=args.lr,
            optimizer_name=args.optimizer,
            criterion_name=args.criterion
        )
        
        all_histories.append(history)
        print(f"Run {run+1} Final Test Acc: {history['test_acc'][-1]:.2f}%")

    print("\n========================================")
    print("       EXPERIMENT SUMMARY")
    print("========================================")
    
    final_accs = [h['test_acc'][-1] for h in all_histories]
    mean_acc = np.mean(final_accs)
    std_acc = np.std(final_accs)
    
    print(f"Model: {args.model}")
    print(f"Runs:  {args.runs}")
    print(f"Mean Final Accuracy: {mean_acc:.2f}%")
    print(f"Std Deviation:       {std_acc:.2f}%")
    print(f"Min / Max Accuracy:  {min(final_accs):.2f}% / {max(final_accs):.2f}%")
    print("========================================")
    
    plot_experiment_results(all_histories, args.model)

if __name__ == '__main__':
    main()