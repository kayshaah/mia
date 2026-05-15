import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_prep import prepare_mia_data

#Define a standard Convolutional Neural Network

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Feature extraction layers
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Classification layers
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 10) # 10 output classes for CIFAR-10
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

#Training Loop

def train_target_model():
    print("Loading data splits...")
    # We only need the target data right now; we ignore the shadow data (_)
    
    target_train_set, target_test_set, _, _ = prepare_mia_data()

    # DataLoaders feed the data to the model in batches
    
    train_loader = DataLoader(target_train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(target_test_set, batch_size=64, shuffle=False)

    # Force PyTorch to use the CPU
    
    device = torch.device("cpu") 
    model = SimpleCNN().to(device)
    
    # Standard loss function for classification and Adam optimizer
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10 # 10 epochs is enough for it to start memorizing the data

    print("\nStarting Target Model Training (This will use your CPU)...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()      # Clear old gradients
            outputs = model(inputs)    # Forward pass
            loss = criterion(outputs, labels) # Calculate error
            loss.backward()            # Backpropagation
            optimizer.step()           # Update weights

            running_loss += loss.item()

        # Print the average loss for this epoch
        avg_loss = running_loss/len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    # 3. Save the trained model weights so we can attack it later
    torch.save(model.state_dict(), 'target_model.pth')
    print("\nTraining complete! Target model saved as 'target_model.pth'.")

if __name__ == "__main__":
    train_target_model()