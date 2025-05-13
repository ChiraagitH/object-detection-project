import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def get_model(num_classes):
    # Pre-trained Faster R-CNN with ResNet50 backbone
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Replace the classifier to match the number of classes (including background)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    return model
