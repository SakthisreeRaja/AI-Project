import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os

# Step 1: Define model (same as training)
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

# Step 2: Get class names from folder
data_path = r"F:\Pokemon ML Training\PokemonData"
classes = sorted(os.listdir(data_path))

# Step 3: Load trained model
model = PokemonCNN(num_classes=len(classes))
model.load_state_dict(torch.load("pokemon_cnn.pth", map_location=torch.device('cpu')))
model.eval()

# Step 4: Load and transform test image
# Step 4: Load and transform test image
image_path = r"F:\Pokemon ML Training\Charizard.jpg"  # Your test image path
image = Image.open(image_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
img_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Step 5: Predict
with torch.no_grad():
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    predicted_class = classes[predicted.item()]

with torch.no_grad():
    output = model(img_tensor)
    probabilities = torch.softmax(output, dim=1)[0]  # Get class probabilities
    confidence, predicted = torch.max(probabilities, 0)
    predicted_class = classes[predicted.item()]
    confidence_percent = confidence.item() * 100

print(f"🔮 Predicted Pokémon: {predicted_class} ({confidence_percent:.2f}% confident)")
