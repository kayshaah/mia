import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_prep import prepare_mia_data
from target_model import SimpleCNN
import numpy as np

def train_and_prep_shadow():
    print("Loading data splits...")
    
    #Creating the shadow data
    
    _, _, shadow_train_set, shadow_test_set = prepare_mia_data()

    train_loader = DataLoader(shadow_train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(shadow_test_set, batch_size=64, shuffle=False)

    device = torch.device("cpu")
    shadow_model = SimpleCNN().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(shadow_model.parameters(), lr=0.001)

    epochs = 10 

    
    print("\nStarting Shadow Model Training...")
    
    for epoch in range(epochs):
        shadow_model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = shadow_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss/len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    
    torch.save(shadow_model.state_dict(), 'shadow_model.pth')
    
    print("\nShadow model saved.")
    
    print("\nExtracting confidence scores for the Attack Model...")
    shadow_model.eval()
    
    attack_X = [] # This will hold the probability vectors
    attack_y = [] # This will hold the labels (1 for Member, 0 for Non-Member)
    
    
    # Softmax converts raw model outputs into percentages (probabilities)
    
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        
        # 1. Process Members (Data it trained on -> Label 1)
        
        for inputs, _ in DataLoader(shadow_train_set, batch_size=64):
            outputs = shadow_model(inputs.to(device))
            probs = softmax(outputs).cpu().numpy()
            attack_X.extend(probs)
            attack_y.extend([1] * len(probs)) 

        
        # 2. Process Non-Members (Data it hasn't seen -> Label 0)
        
        for inputs, _ in DataLoader(shadow_test_set, batch_size=64):
            outputs = shadow_model(inputs.to(device))
            probs = softmax(outputs).cpu().numpy()
            attack_X.extend(probs)
            attack_y.extend([0] * len(probs)) 

    
    # Save the extracted probabilities so we can build a simple classifier next
    np.save('attack_X.npy', np.array(attack_X))
    np.save('attack_y.npy', np.array(attack_y))
    print("Attack dataset saved (attack_X.npy, attack_y.npy).")

if __name__ == "__main__":
    train_and_prep_shadow()