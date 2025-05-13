import torch
from torch.utils.data import DataLoader
from dataset import CustomDataset
from model import get_model
import torchvision.transforms as T
import os

def get_transform():
    return T.Compose([T.ToTensor()])

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset paths
    train_dataset = CustomDataset(
        root_dir="C:\Users\chiraag\object-detection-project\data\COCO 128.v2-640x640.coco\train",  
        annotation_file="C:\Users\chiraag\object-detection-project\data\COCO 128.v2-640x640.coco\train\_annotations.coco.json",
        transforms=get_transform()
    )

    val_dataset = CustomDataset(
        root_dir="C:\Users\chiraag\object-detection-project\data\COCO 128.v2-640x640.coco\valid",
        annotation_file="C:\Users\chiraag\object-detection-project\data\COCO 128.v2-640x640.coco\valid\_annotations.coco.json"
        transforms=get_transform()
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    num_classes = len(train_dataset.image_id_to_annotations) + 1  # +1 for background
    model = get_model(num_classes).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    for epoch in range(5):
        model.train()
        for images, targets in train_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1} Loss: {losses.item()}")

    os.makedirs("outputs/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "outputs/checkpoints/final_model.pth")

if __name__ == "__main__":
    main()
