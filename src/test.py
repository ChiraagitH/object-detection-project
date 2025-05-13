import torch
import torchvision.transforms.functional as F
import cv2
import matplotlib.pyplot as plt
from model import get_resnet_backbone

def test_model(model, image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_tensor = F.to_tensor(image).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor)

    plt.imshow(image)
    plt.title(f"Predicted Labels: {prediction[0]['labels'].cpu().numpy()}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    model = get_resnet_backbone(num_classes=2)  # Or load trained model
    model.load_state_dict(torch.load("saved_model.pth", map_location=torch.device("cpu")))  # Use your path
    test_model(model, "test_image.jpg")
