import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(28*28, 64)  
        self.fc2 = nn.Linear(64, 10)     
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)  
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  
])

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        f.read(16)  
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(-1, 28 * 28) / 255.0 

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        f.read(8) 
        return np.frombuffer(f.read(), dtype=np.uint8)
    
train_images = load_mnist_images('train-images.idx3-ubyte')
train_labels = load_mnist_labels('train-labels.idx1-ubyte')
test_images = load_mnist_images('t10k-images.idx3-ubyte')
test_labels = load_mnist_labels('t10k-labels.idx1-ubyte')

train_images_tensor = torch.tensor(train_images, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
test_images_tensor = torch.tensor(test_images, dtype=torch.float32)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = NN()
criterion = nn.NLLLoss()  
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.view(-1, 28*28)  
            optimizer.zero_grad() 
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward() 
            optimizer.step() 
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1} complete')

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_images = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 28*28)  # Flatten the images
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_images.extend(images.view(-1, 28, 28).numpy())
            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.numpy())

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    num_samples = 5
    indices = np.random.choice(len(all_images), num_samples, replace=False)

    plt.figure(figsize=(10, 5))
    for i, idx in enumerate(indices):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(all_images[idx], cmap='gray')
        plt.title(f'Label: {all_labels[idx]}\nPred: {all_preds[idx]}')
        plt.axis('off')
    plt.show()

train(model, train_loader, criterion, optimizer, epochs=10)
evaluate(model, test_loader)
