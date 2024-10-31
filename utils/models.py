import torch
import torch.nn as nn
from typing import List
from torchvision import models
import numpy as np
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class UncertaintyResNetPretrained(nn.Module):
    """
    A ResNet model with dropout layers added to specified submodules for 
    uncertainty estimation.
    """  

    def __init__(self, num_classes: int, dropout_rate: float = 0.25, pretrained_weights = True):
        """
        Initialize the model with the specified number of output classes and 
        dropout rate. Optionally use pretrained weights.
        
        Args:
            num_classes (int): Number of output classes for the classification.
            dropout_rate (float): Dropout rate to use for dropout layers.
            pretrained_weights (bool): Whether to use pretrained weights or not.
        """  

        super().__init__()
        
        self.num_classes = num_classes
        model_weights = ResNet18_Weights.DEFAULT if pretrained_weights else None
        self.base_model = resnet18(weights=model_weights)
        self.base_model.fc = nn.Linear(in_features=self.base_model.fc.in_features, out_features=num_classes)

        # Modify the first convolutional layer to accept single-channel input (e.g. grayscale images)
        w = self.base_model.conv1.weight
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=2, bias=False)
        self.base_model.conv1.weight = nn.Parameter(torch.mean(w, dim=1, keepdim=True))

        # Define dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Apply dropout to specific layers
        self.apply_dropout(self.base_model, ['relu', 'bn2', 'downsample'])

    def apply_dropout(self, module, target_layers):
        """
        Recursively applies dropout to specified layers in the module.
        """
        for name, submodule in module.named_children():
            # Apply dropout to the target layers
            if any(target in name for target in target_layers):
                setattr(module, name, nn.Sequential(submodule, self.dropout))
            
            # Recursively go through the child modules
            self.apply_dropout(submodule, target_layers)

    def forward(self, x):
        return self.base_model(x)
    

    @torch.no_grad()
    def get_prediction_stats(self, x, n_samples, batch_size):
        """
        Calculate the mean, standard deviation, and raw predictions across 
        Monte Carlo (MC) dropout samples.
        
        Args:
            x (torch.Tensor): Input tensor.
            n_samples (int): Number of Monte Carlo samples.
            batch_size (int): Size of the input batch.
        
        Returns:
            tuple: A tuple containing the mean predictions, standard deviation, 
                and raw predictions tensor.
        """
        
        preds = torch.zeros((batch_size, self.num_classes, n_samples))
            
        for i in range(n_samples):
            logits = self(x)
            probs = logits
            preds[:,:,i] = probs
        
        mean = torch.mean(preds,dim=2)
        std = torch.std(preds,dim=2)
        
        return mean, std, preds
    

    @torch.no_grad()
    def get_features(self,x):
        """
        Extract high-level features from the CNN output, from the FC layer.

        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: A tensor containing the FC layer output.
        """
        def forward_imp(self,x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            xf = self.fc(x)

            return x, xf
        
        return forward_imp(self.base_model,x)