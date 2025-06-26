import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


torch.manual_seed(42)

#  Step 1: Prepare MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensors (0-1 range)
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std normalization
])

# Download/Load MNIST
train_data = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)
test_data = datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

#Step 2: Define the Neural Network
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()  
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),  
            nn.ReLU(),              
            nn.Linear(512, 256),    
            nn.ReLU(),
            nn.Linear(256, 10)      
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = MNISTNet()

#  Step 3: Training Setup 
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)  

# Track training progress
train_loss_history = []
train_acc_history = []

# Step 4: Training Loop 
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Epoch statistics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    train_loss_history.append(epoch_loss)
    train_acc_history.append(epoch_acc)
    
    print(f'Epoch {epoch+1}/{epochs}, '
          f'Loss: {epoch_loss:.4f}, '
          f'Accuracy: {epoch_acc:.2f}%')

# Step 5: Evaluation 
model.eval()  
test_correct = 0
test_total = 0

with torch.no_grad():  # Disable gradient calculation
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

print(f'\nTest Accuracy: {100 * test_correct / test_total:.2f}%')

# Step 6: Visualization 
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc_history, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()

#  Save the Model
torch.save(model.state_dict(), 'mnist_model.pth')