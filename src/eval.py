import torch
from sklearn.metrics import precision_score, recall_score

def evaluate(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = [image.to(device) for image in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                preds = output['labels'].cpu().numpy()
                labels = target['labels'].cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels)

    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")

