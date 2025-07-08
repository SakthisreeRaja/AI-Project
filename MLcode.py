# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

# # 1. Define transforms (resize, normalize)
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor()
# ])

# # 2. Load dataset (train and validation)
# data_path = r"F:\Pokemon ML Training\PokemonData"
# train_set = datasets.ImageFolder(root=data_path, transform=transform)
# train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

# # 3. Define CNN model
# class PokemonCNN(nn.Module):
#     def __init__(self, num_classes):
#         super(PokemonCNN, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Flatten(),
#             nn.Linear(64 * 32 * 32, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_classes)
#         )

#     def forward(self, x):
#         return self.model(x)

# # 4. Initialize model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = PokemonCNN(num_classes=len(train_set.classes)).to(device)

# # 5. Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # 6. Training loop
# for epoch in range(10):
#     model.train()
#     total_loss = 0
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)

#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# # 7. Save the model (optional)
# torch.save(model.state_dict(), "pokemon_cnn.pth")


#Retrain code
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Define transforms (resize, normalize)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# 2. Load dataset (train and validation)
data_path = r"F:\Pokemon ML Training\PokemonData"
train_set = datasets.ImageFolder(root=data_path, transform=transform)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

# 3. Define CNN model
class PokemonCNN(nn.Module):
    def __init__(self, num_classes):
        super(PokemonCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# 4. Initialize and Load previous model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PokemonCNN(num_classes=len(train_set.classes)).to(device)

# 🔁 Load old weights for retraining
model.load_state_dict(torch.load("pokemon_cnn.pth", map_location=device))
model.train()  # set to training mode again

# 5. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Retraining loop
for epoch in range(2):  # add more if needed
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"🔁 Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 7. Save updated weights
torch.save(model.state_dict(), "pokemon_cnn.pth")
print("✅ Retrained model saved successfully!")

