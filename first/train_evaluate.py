import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from tqdm import tqdm 

def train_model(model, train_loader, val_loader, test_loader, device, epochs=10, lr=0.001):
    """
    Training Loop with 3-Split Logic:
    1. Train on TRAIN set.
    2. Evaluate on VAL set -> Used by Scheduler to adjust LR (Improve Model).
    3. Evaluate on TEST set -> Used only for plotting (Monitor Overfitting).
    """
    criterion = nn.CrossEntropyLoss()
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=lr)
    
    # Scheduler: Reduces LR if Validation Loss stops decreasing
    # Note: 'verbose' argument removed for compatibility with PyTorch 2.4+
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3
    )

    # Lists to store metrics for plotting
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'test_loss': [], 'test_acc': []
    }

    print(f"Training on {device} for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(epochs):
        # --- 1. TRAINING PHASE ---
        model.train() 
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, leave=False)
        
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        # --- 2. VALIDATION PHASE (Used to Improve Model) ---
        val_loss, val_acc = evaluate_model(model, val_loader, device, criterion)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Step the scheduler based on Validation Loss
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"Scheduler: Reducing learning rate from {old_lr} to {new_lr}")

        # --- 3. TESTING PHASE (Observation Only) ---
        test_loss, test_acc = evaluate_model(model, test_loader, device, criterion)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print(f"Epoch {epoch+1}: Train Loss={epoch_loss:.4f} | Val Loss={val_loss:.4f} | Test Loss={test_loss:.4f}")
        print(f"          Train Acc ={epoch_acc:.2f}%  | Val Acc ={val_acc:.2f}%  | Test Acc ={test_acc:.2f}%")

    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f} seconds.")
    
    return history

def evaluate_model(model, loader, device, criterion):
    model.eval() 
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def plot_results(history, model_name):
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Loss (Train vs Val vs Test)
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss', color='blue')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss', color='green', linestyle='--')
    plt.plot(epochs_range, history['test_loss'], label='Testing Loss', color='red', linestyle=':')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Accuracy (Train vs Val vs Test)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Training Acc', color='blue')
    plt.plot(epochs_range, history['val_acc'], label='Validation Acc', color='green', linestyle='--')
    plt.plot(epochs_range, history['test_acc'], label='Testing Acc', color='red', linestyle=':')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_results.png')
    print(f"Plot saved as {model_name}_results.png")
    plt.show()