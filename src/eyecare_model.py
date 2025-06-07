import torch.nn as nn
from torchvision import models
import torch # Added for the example usage in __main__

class TransferLearningModel(nn.Module):
    def __init__(self, num_classes, model_name="resnet18", pretrained=True):
        super(TransferLearningModel, self).__init__()
        self.model_name = model_name

        if model_name == "resnet18":
            # Use weights=models.ResNet18_Weights.DEFAULT for modern PyTorch
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == "alexnet":
            # Use weights=models.AlexNet_Weights.DEFAULT for modern PyTorch
            self.model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT if pretrained else None)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        else:
            raise ValueError(f"Model '{model_name}' not supported. Choose 'resnet18' or 'alexnet'.")

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    # Example usage:
    num_classes_example = 3 # Adjust based on your actual number of classes
    print(f"Testing ResNet18 model with {num_classes_example} classes:")
    model_resnet = TransferLearningModel(num_classes=num_classes_example, model_name="resnet18", pretrained=True)
    print(model_resnet)

    dummy_input = torch.randn(1, 3, 224, 224) # Batch size 1, 3 channels, 224x224 image
    output = model_resnet(dummy_input)
    print(f"Output shape for ResNet18: {output.shape}")

    print("\nTesting AlexNet model:")
    model_alexnet = TransferLearningModel(num_classes=num_classes_example, model_name="alexnet", pretrained=True)
    print(model_alexnet)
    output = model_alexnet(dummy_input)
    print(f"Output shape for AlexNet: {output.shape}")