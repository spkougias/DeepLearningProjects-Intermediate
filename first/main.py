import torch
import argparse
import sys
import os

# --- IMPORTS EXPLANATION ---
# The error "ImportError: attempted relative import..." happens when using dots (from .utils)
# in the main script. We must use absolute imports (from utils).

from utils import get_cifar10_loaders
from models import SimpleMLP, ComplexMLP, SimpleCNN, ComplexCNN, TransferLearningModel
from train_evaluate import train_model, plot_results



def get_user_input(prompt, default_value, value_type):
    """Helper function to get input with a default value."""
    user_input = input(f"{prompt} [default: {default_value}]: ").strip()
    if user_input == "":
        return default_value
    try:
        return value_type(user_input)
    except ValueError:
        print(f"Invalid input. Using default: {default_value}")
        return default_value

def main():
    parser = argparse.ArgumentParser(description="Deep Learning Assignment - CIFAR10")
    parser.add_argument('--model', type=str, default='simple_cnn', 
                        choices=['simple_mlp', 'complex_mlp', 'simple_cnn', 'complex_cnn', 'transfer'],
                        help='Choose the architecture to train')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--no_augment', action='store_true', help='Disable data augmentation')
    parser.add_argument('--interactive', action='store_true', help='Prompt for parameters at runtime')
    
    args = parser.parse_args()

    # ==========================================
    # INTERACTIVE MODE
    # ==========================================
    print("========================================")
    print("       CONFIGURATION SETTINGS")
    print("========================================")
    print(f"Current Model:    {args.model}")
    print(f"Current Epochs:   {args.epochs}")
    print(f"Current Batch:    {args.batch_size}")
    print(f"Current LR:       {args.lr}")
    print("----------------------------------------")
    
    change_settings = input("Do you want to change these settings? (y/n) [default: n]: ").strip().lower()
    
    if change_settings == 'y':
        print("\n--- Enter new values (press Enter to keep default) ---")
        
        print("Available models: simple_mlp, complex_mlp, simple_cnn, complex_cnn, transfer")
        model_input = input(f"Model [{args.model}]: ").strip()
        if model_input in ['simple_mlp', 'complex_mlp', 'simple_cnn', 'complex_cnn', 'transfer']:
            args.model = model_input
            
        args.epochs = get_user_input("Epochs", args.epochs, int)
        args.batch_size = get_user_input("Batch Size", args.batch_size, int)
        args.lr = get_user_input("Learning Rate", args.lr, float)
        
        print("\nNew Settings confirmed.")
    else:
        print("Using default settings.")

    # 1. Setup Device
    print("\n------------------------------------------------")
    print(f"PyTorch Version: {torch.__version__}")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ SUCCESS: GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("⚠️  WARNING: GPU NOT detected. Using CPU.")
        print("------------------------------------------------")

    # 2. Load Data
    print("\nLoading Data (Split: Train / Val / Test)...")
    augment = not args.no_augment
    # Returns 3 loaders now
    train_loader, val_loader, test_loader, classes = get_cifar10_loaders(
        batch_size=args.batch_size, augment=augment
    )

    # 3. Initialize Model
    print(f"Initializing {args.model}...")
    if args.model == 'simple_mlp':
        model = SimpleMLP()
    elif args.model == 'complex_mlp':
        model = ComplexMLP()
    elif args.model == 'simple_cnn':
        model = SimpleCNN()
    elif args.model == 'complex_cnn':
        model = ComplexCNN()
    elif args.model == 'transfer':
        model = TransferLearningModel()
    
    model = model.to(device)

    # 4. Train
    print(f"Starting training with Learning Rate: {args.lr}, Batch Size: {args.batch_size}")
    
    # The history dictionary contains all 3 metrics
    history = train_model(
        model, train_loader, val_loader, test_loader, device, 
        epochs=args.epochs, lr=args.lr
    )

    # 5. Plot Results
    plot_results(history, args.model)
    
    # Final Result Print
    print("================================================")
    print(f"FINAL RESULTS for {args.model}:")
    print(f"Final Test Loss:     {history['test_loss'][-1]:.4f}")
    print(f"Final Test Accuracy: {history['test_acc'][-1]:.2f}%")
    print("================================================")

if __name__ == '__main__':
    main()