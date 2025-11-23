import torch
import argparse
import sys
import os
from utils import get_cifar10_loaders
from train_evaluate import train_model, plot_results, show_confusion_matrix
from models import initialize_model


def main():
    parser = argparse.ArgumentParser(description="Deep Learning Assignment - CIFAR10")
    
    parser.add_argument('--model', type=str, default='simple_cnn', 
                        choices=['simple_mlp', 'complex_mlp', 'cnn', 'transfer'], 
                        help='Choose the architecture to train')
    parser.add_argument('--criterion', type=str, default='cross_entropy', 
                        choices=['cross_entropy', 'label_smoothing', 'hinge_loss'], 
                        help='Choose the criterion used in training')
    parser.add_argument('--optimizer', type=str, default='adam', 
                        choices=['adam', 'adamw', 'sgd'], 
                        help='Choose the optimizer used in training')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'leaky_relu', 'sigmoid'],
                        help='Choose the activation function')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (default: 0.5). Set to 0 to disable.')
    
    args = parser.parse_args()

    # 1. Setup Device
    print("------------------------------------------------")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"SUCCESS: GPU found: {torch.cuda.get_device_name(0)}")
        
    else:
        device = torch.device('cpu')
        print("WARNING: GPU NOT found. Using CPU.")
    print("------------------------------------------------")

    print("\nLoading Data...\n")
    print("##########################################################################################################")
    print(f"> Settings: Epochs: {args.epochs} | Learning rate: {args.lr} | Batch Size: {args.batch_size} | Dropout: {args.dropout} | Augment: {args.augment}")
    print(f"> Optimizer: {args.optimizer}")
    print(f"> Criterion: {args.criterion}")
    print(f"> Activation: {args.activation}")
    print("##########################################################################################################\n")
    

    train_loader, val_loader, test_loader, classes = get_cifar10_loaders(
        batch_size=args.batch_size, augment= args.augment
    )


    # init model
    print(f"Initializing {args.model}...")
    model = initialize_model(args.model, args.dropout, device)
    
    history = train_model(
        model, train_loader, val_loader, test_loader, device, 
        epochs=args.epochs, lr=args.lr
    )

    # Plot Results
    plot_results(history, args.model)
    
    # Final Result Print
    print("================================================")
    print(f"FINAL RESULTS for {args.model}:")
    print(f"Final Test Loss:     {history['test_loss'][-1]:.4f}")
    print(f"Final Test Accuracy: {history['test_acc'][-1]:.2f}%")
    print("================================================")
    show_confusion_matrix(model, test_loader, device, classes, args.model)

if __name__ == '__main__':
    main()