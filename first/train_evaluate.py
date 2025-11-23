import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def train_model(model, train_loader, val_loader, test_loader, device, epochs=10, lr=0.001, 
                optimizer_name='adam', criterion_name='cross_entropy'):
    # criterion
    match criterion_name:
        case 'cross_entropy':
            criterion = nn.CrossEntropyLoss()
        case 'label_smoothing':
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        case 'hinge_loss':
            criterion = nn.MultiMarginLoss()
        case _:
            raise ValueError(f"404 Criterion not found")

    # Optimizer
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    
    match optimizer_name:
        case 'adam':
            optimizer = optim.Adam(params_to_update, lr=lr)
        case 'adamw':
            optimizer = optim.AdamW(params_to_update, lr=lr)
        case 'sgd':
            optimizer = optim.SGD(params_to_update, lr=lr, momentum=0.9)
        case _:
            raise ValueError(f"404 Optimizer not found")
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'test_loss': [], 'test_acc': []
    }

    print(f"Training on {device} for {epochs} epochs")
    print(f"Optimizer: {optimizer_name} | Criterion: {criterion_name}")

    # train loop
    start_time = time.time()
    for epoch in range(epochs):
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
        
        # validation
        val_loss, val_acc = evaluate_model(model, val_loader, device, criterion)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f">>Scheduler: Learning rate changed from {old_lr} to {new_lr}")

        # testing
        test_loss, test_acc = evaluate_model(model, test_loader, device, criterion)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f"Epoch {epoch+1}:====================================================================================")
        print(f"///////////////| Train Loss={epoch_loss:.4f} | Val Loss={val_loss:.4f} | Test Loss={test_loss:.4f}")
        print("///////////////|")
        print(f"///////////////| Train Acc ={epoch_acc:.2f}% | Val Acc ={val_acc:.2f}% | Test Acc ={test_acc:.2f}%")
    
    total_time = time.time() - start_time
    print(f"Training took {total_time:.2f} seconds")
    
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
    
    # Plot_1: Loss (Train vs Val vs Test)
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss', color='blue')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss', color='green', linestyle='--')
    plt.plot(epochs_range, history['test_loss'], label='Testing Loss', color='red', linestyle=':')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot_2: Accuracy (Train vs Val vs Test)
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
    
def show_confusion_matrix(model, loader, device, classes, model_name):
    print("\nConfusion Matrix...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating for Matrix"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    
    # Plot with darker blues for higher numbers
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', values_format='d', ax=ax)
    
    plt.title(f'Confusion Matrix: {model_name}')
    plt.tight_layout()
    filename = f'{model_name}_confusion_matrix.png'
    plt.savefig(filename)
    print(f"Confusion Matrix saved as {filename}")
    plt.show()