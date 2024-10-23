import torch
import torch.nn as nn
from typing import List
from torchvision import models
import numpy as np
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class UncertaintyResNetPretrained(nn.Module):
    def __init__(self, num_classes: int, dropout_rate: float = 0.25, pretrained_weights = True):
        super().__init__()
        
        self.num_classes = num_classes
        
        if pretrained_weights:
            model_weights = ResNet18_Weights.DEFAULT
            self.base_model = resnet18(weights=model_weights)
            self.base_model.fc = torch.nn.Linear(in_features=self.base_model.fc.in_features,out_features=num_classes)
        else:
            self.base_model = resnet18()
            self.base_model.fc = torch.nn.Linear(in_features=self.base_model.fc.in_features,out_features=num_classes)
        
        w = self.base_model.conv1.weight
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=2, bias=False)
        self.base_model.conv1.weight = nn.Parameter(torch.mean(w, dim=1, keepdim=True))
        
        #net = net.to(dev)
        
        # Adding dropout layers inside residual blocks
        #self.dropouts = nn.ModuleList([nn.Dropout3d(p=dropout_rate) for _ in range(17)])
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # for name, module in self.base_model.named_modules():
        #     if name == '':
        #         #print(name)
        #         for subname, submodule in module.named_children():
        #             #print(subname)
        #             for subname2, submodule2 in submodule.named_children():
        #                 for subname3, submodule3 in submodule2.named_children():
        #                     for subname4, submodule4 in submodule3.named_children():
        #                         if '1' in subname4 and 'downsample' in subname3:
        #                             setattr(submodule3, subname4,nn.Sequential(submodule4, self.dropout))

        for name, module in self.base_model.named_modules():
            if name == '':
                for subname, submodule in module.named_children():
                    for subname2, submodule2 in submodule.named_children():
                        for subname3, submodule3 in submodule2.named_children():
                            if 'relu' in subname3:
                                setattr(submodule2, subname3,nn.Sequential(submodule3, self.dropout))
                            
        for name, module in self.base_model.named_modules():
            #print(f"name -{name}-")
            if name == '':
                for subname, submodule in module.named_children():
                    #print(f" subname: {subname}")
                    if 'relu' in subname:
                        setattr(module, subname,nn.Sequential(submodule, self.dropout))

#         # List of modules to modify
#         modules_to_modify = []
        
#         # Add dropout after the ReLU layers
#         for name, module in self.base_model.named_modules():
#             if 'relu' in name or 'bn2' in name or 'downsample.1' in name:
#                 modules_to_modify.append((name, module))

# #         # Perform modifications outside the loop
#         for name, module in modules_to_modify:
#             setattr(self.base_model, name, nn.Sequential(module, self.dropout))
        
        #self.n_samples = n_samples
        #print(self.base_model.named_modules())
        
    def forward(self, x):
        x = self.base_model(x)
        return x
    
    @torch.no_grad()
    def get_prediction_stats(self, x: torch.Tensor, n_samples, batch_size) -> torch.Tensor:
        """Calculate the mean prediction across all MC drops."""
        
        preds = torch.zeros((batch_size, self.num_classes, n_samples))
            
        for i in range(n_samples):
            logits = self(x)
            #probs = torch.softmax(logits, dim=-1)
            probs = logits
            preds[:,:,i] = probs
        
        mean = torch.mean(preds,dim=2)
        std = torch.std(preds,dim=2)
        
        return mean, std, preds

    @torch.no_grad()
    def get_stddev_prediction(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the standard deviation of predictions across all MC drops."""
        stds: List[torch.Tensor] = []
        for _ in range(self.n_samples):
            logits, _ = self(x)
            probs = torch.softmax(logits, dim=-1)
            variance = ((probs - self.get_mean_prediction(x)) ** 2).sum(dim=-1, keepdim=True)
            stds.append(variance.sqrt())

        return torch.cat(stds, dim=-1)
    
    @torch.no_grad()
    def get_features(self,x):
    
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

        

    
    
class UncertaintyResNetPretrainedBNnoTrack(nn.Module):
    def __init__(self, num_classes: int, dropout_rate: float = 0.25, pretrained_weights = True):
        super().__init__()
        
        self.num_classes = num_classes
        
        if pretrained_weights:
            model_weights = ResNet18_Weights.DEFAULT
            self.base_model = resnet18(weights=model_weights)
            self.base_model.fc = torch.nn.Linear(in_features=self.base_model.fc.in_features,out_features=num_classes)
        else:
            self.base_model = resnet18()
            self.base_model.fc = torch.nn.Linear(in_features=self.base_model.fc.in_features,out_features=num_classes)
        
        #net = net.to(dev)
        
        # Iterate over the model's modules
        for module in self.base_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False
        
        # Adding dropout layers inside residual blocks
        #self.dropouts = nn.ModuleList([nn.Dropout3d(p=dropout_rate) for _ in range(17)])
        self.dropout = nn.Dropout(p=dropout_rate)
        
        for name, module in self.base_model.named_modules():
            if name == '':
                #print(name)
                for subname, submodule in module.named_children():
                    #print(subname)
                    for subname2, submodule2 in submodule.named_children():
                        for subname3, submodule3 in submodule2.named_children():
                            for subname4, submodule4 in submodule3.named_children():
                                if '1' in subname4 and 'downsample' in subname3:
                                    setattr(submodule3, subname4,nn.Sequential(submodule4, self.dropout))

        for name, module in self.base_model.named_modules():
            if name == '':
                for subname, submodule in module.named_children():
                    for subname2, submodule2 in submodule.named_children():
                        for subname3, submodule3 in submodule2.named_children():
                            if 'relu' in subname3 or 'bn2' in subname3:
                                setattr(submodule2, subname3,nn.Sequential(submodule3, self.dropout))
                            
        for name, module in self.base_model.named_modules():
            #print(f"name -{name}-")
            if name == '':
                for subname, submodule in module.named_children():
                    #print(f" subname: {subname}")
                    if 'relu' in subname:
                        setattr(module, subname,nn.Sequential(submodule, self.dropout))
        
    def forward(self, x):
        x = self.base_model(x)
        return x
    
    @torch.no_grad()
    def get_prediction_stats(self, x: torch.Tensor, n_samples, batch_size) -> torch.Tensor:
        """Calculate the mean prediction across all MC drops."""
        
        preds = torch.zeros((batch_size, self.num_classes, n_samples))
            
        for i in range(n_samples):
            logits = self(x)
            #probs = torch.softmax(logits, dim=-1)
            probs = logits
            preds[:,:,i] = probs
        
        mean = torch.mean(preds,dim=2)
        std = torch.std(preds,dim=2)
        
        return mean, std, preds

class UncertaintyResNetPretrainedFCdrop(nn.Module):
    def __init__(self, num_classes: int, dropout_rate: float = 0.25, pretrained_weights = True):
        super().__init__()
        
        self.num_classes = num_classes
        
        if pretrained_weights:
            model_weights = ResNet18_Weights.DEFAULT
            self.base_model = resnet18(weights=model_weights)
            self.base_model.fc = torch.nn.Linear(in_features=self.base_model.fc.in_features,out_features=num_classes)
        else:
            self.base_model = resnet18()
            self.base_model.fc = torch.nn.Linear(in_features=self.base_model.fc.in_features,out_features=num_classes)
        
        #net = net.to(dev)
        
        # Adding dropout layers inside residual blocks
        #self.dropouts = nn.ModuleList([nn.Dropout3d(p=dropout_rate) for _ in range(17)])
        self.dropout = nn.Dropout(p=dropout_rate)
                            
        for name, module in self.base_model.named_modules():
            #print(f"name -{name}-")
            if name == '':
                for subname, submodule in module.named_children():
                    #print(f" subname: {subname}")
                    if 'avgpool' in subname:
                        setattr(module, subname,nn.Sequential(submodule, self.dropout))

        
    def forward(self, x):
        x = self.base_model(x)
        return x
    
    
    @torch.no_grad()
    def get_prediction_stats(self, x: torch.Tensor, n_samples, batch_size) -> torch.Tensor:
        """Calculate the mean prediction across all MC drops."""
        
#         def forward_pass_until_fc(model, input_tensor):
#             x = input_tensor
#             for module in model.base_model.children():
#                 if isinstance(module, nn.Sequential):
#                     for layer in module.children():
#                         x = layer(x)
#                 else:
#                     x = module(x)
#                 if isinstance(module, nn.AdaptiveAvgPool2d):
#                     break
#             return x.view(-1, 512)  # flatten the output to feed into the FC layer
        
        def forward_pass_all(model, input_tensor):
            x = input_tensor
            
            for module in model.base_model.children():
                #print(module.__class__.__name__)
#                 if isinstance(module, nn.Sequential):
                    
#                     for layer in module.children():
#                         if isinstance(layer, nn.Sequential):
#                             for layer2 in layer.childen():
#                                 x = layer2(x)
#                         else:
#                             x = layer(x)
#                 else:
                if isinstance(module, nn.Linear):
                    x = module(x.view(-1,512))
                else:
                    x = module(x)
            return x
    
        def forward_pass_until_avgpool(model, input_tensor):
            x = input_tensor
            a=0
            for module in model.base_model.children():
                #print(module.__class__.__name__)
                if isinstance(module, nn.Sequential):
                    a+=1
                    for layer in module.children():
                        if isinstance(layer, nn.Sequential):
                            for layer2 in layer.childen():
                                x = layer2(x)
                        else:
                            x = layer(x)
                else:
                    x = module(x)
                if a==4:
                    break
#                 if isinstance(module, nn.AdaptiveAvgPool2d):
#                     return x
            return x

        
        def forward_pass_from_avgpool(model, input_tensor):
            x = input_tensor
            a=0
            for module in model.base_model.children():
                #print(module.__class__.__name__)
                if isinstance(module, nn.Sequential):
                    a+=1
                    if a>4:
                        for layer in module.children():
                            x = layer(x)
                        if a==5:
                            x = x.view(-1, 512)
                            #print(layer)
                elif a>4:
                    x = module(x)

#                 if isinstance(module, nn.AdaptiveAvgPool2d):
#                     x = module(x)
#                     a=1
            return x


        def forward_avgpool(model,x):
        # a dict to store the activations
            activation = {}
            def getActivation(name):
              # the hook signature
              def hook(model, input, output):
                activation[name] = output.detach()
              return hook
        
            
            # register forward hooks on the layers of choice
            h1 = model.avgpool.register_forward_hook(getActivation('avgpool'))
            
            # forward pass -- getting the outputs
            out = model(x)

            # detach the hooks
            h1.remove()
            
            print(activation.keys())
            
            return activation['avgpool']
        
        def forward_impl(self, x):
            # See note [TorchScript super()]
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            #x = self.avgpool(x)
            #x = torch.flatten(x, 1)
            #x = self.fc(x)

            return x
        
        def from_avg(self,x):
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            
            return x

        #up_model = nn.Sequential(*list(list(self.children())[0].children()))
        up_model = nn.Sequential(*list(self.base_model.children()))
        down_model = nn.Sequential(*list(list(self.children())[0].children())[-2:])
        
        preds = torch.zeros((batch_size, self.num_classes, n_samples))
        
        #model = self.base_model
        #delattr(model,'fc')
        
        x_base = forward_impl(self.base_model, x)
        #print(x_base.size())
        #print(self.base_model)
        
        for i in range(n_samples):
            #logits = forward_impl(self.base_model, x)
            logits = from_avg(self.base_model,x_base)
            #logits = self.base_model.fc(x_base)
            #logits = self(x)
            #print(logits.size())
            #probs = torch.softmax(logits, dim=-1)
            probs = logits
            preds[:,:,i] = probs
        
        mean = torch.mean(preds,dim=2)
        std = torch.std(preds,dim=2)
        
        return mean, std, preds
    

class UncertaintyResNetPretrainedLayer4drop(nn.Module):
    def __init__(self, num_classes: int, dropout_rate: float = 0.25, pretrained_weights = True):
        super().__init__()
        
        self.num_classes = num_classes
        
        if pretrained_weights:
            model_weights = ResNet18_Weights.DEFAULT
            self.base_model = resnet18(weights=model_weights)
            self.base_model.fc = torch.nn.Linear(in_features=self.base_model.fc.in_features,out_features=num_classes)
        else:
            self.base_model = resnet18()
            self.base_model.fc = torch.nn.Linear(in_features=self.base_model.fc.in_features,out_features=num_classes)
        
        #net = net.to(dev)
        
        # Adding dropout layers inside residual blocks
        #self.dropouts = nn.ModuleList([nn.Dropout3d(p=dropout_rate) for _ in range(17)])
        self.dropout = nn.Dropout(p=dropout_rate)
        
        for name, module in self.base_model.named_modules():
            if name == '':
                #print(name)
                for subname, submodule in module.named_children():
                    #print(subname)
                    for subname2, submodule2 in submodule.named_children():
                        for subname3, submodule3 in submodule2.named_children():
                            for subname4, submodule4 in submodule3.named_children():
                                if ('1' in subname4 and 'downsample' in subname3) and 'layer4' in subname:
                                    setattr(submodule3, subname4,nn.Sequential(submodule4, self.dropout))

        for name, module in self.base_model.named_modules():
            if name == '':
                for subname, submodule in module.named_children():
                    for subname2, submodule2 in submodule.named_children():
                        for subname3, submodule3 in submodule2.named_children():
                            if ('relu' in subname3 or 'bn2' in subname3) and 'layer4' in subname:
                                setattr(submodule2, subname3,nn.Sequential(submodule3, self.dropout))
                            
        
    def forward(self, x):
        x = self.base_model(x)
        return x
    
    
    @torch.no_grad()
    def get_prediction_stats(self, x: torch.Tensor, n_samples, batch_size) -> torch.Tensor:
        """Calculate the mean prediction across all MC drops."""
        
        
        def forward_impl(self, x):
            # See note [TorchScript super()]
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            #x = self.layer4(x)

            #x = self.avgpool(x)
            #x = torch.flatten(x, 1)
            #x = self.fc(x)

            return x
        
        def from_avg(self,x):
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            
            return x

        #up_model = nn.Sequential(*list(list(self.children())[0].children()))
        up_model = nn.Sequential(*list(self.base_model.children()))
        down_model = nn.Sequential(*list(list(self.children())[0].children())[-2:])
        
        preds = torch.zeros((batch_size, self.num_classes, n_samples))
        
        #model = self.base_model
        #delattr(model,'fc')
        
        x_base = forward_impl(self.base_model, x)
        #print(x_base.size())
        #print(self.base_model)
        
        for i in range(n_samples):
            #logits = forward_impl(self.base_model, x)
            logits = from_avg(self.base_model,x_base)
            #logits = self.base_model.fc(x_base)
            #logits = self(x)
            #print(logits.size())
            #probs = torch.softmax(logits, dim=-1)
            probs = logits
            preds[:,:,i] = probs
        
        mean = torch.mean(preds,dim=2)
        std = torch.std(preds,dim=2)
        
        return mean, std, preds
    
    
class UncertaintyResNetPretrainedLayer3drop(nn.Module):
    def __init__(self, num_classes: int, dropout_rate: float = 0.25, pretrained_weights = True):
        super().__init__()
        
        self.num_classes = num_classes
        
        if pretrained_weights:
            model_weights = ResNet18_Weights.DEFAULT
            self.base_model = resnet18(weights=model_weights)
            self.base_model.fc = torch.nn.Linear(in_features=self.base_model.fc.in_features,out_features=num_classes)
        else:
            self.base_model = resnet18()
            self.base_model.fc = torch.nn.Linear(in_features=self.base_model.fc.in_features,out_features=num_classes)
        
        #net = net.to(dev)
        
        # Adding dropout layers inside residual blocks
        #self.dropouts = nn.ModuleList([nn.Dropout3d(p=dropout_rate) for _ in range(17)])
        self.dropout = nn.Dropout(p=dropout_rate)
        
        for name, module in self.base_model.named_modules():
            if name == '':
                #print(name)
                for subname, submodule in module.named_children():
                    #print(subname)
                    for subname2, submodule2 in submodule.named_children():
                        for subname3, submodule3 in submodule2.named_children():
                            for subname4, submodule4 in submodule3.named_children():
                                if ('1' in subname4 and 'downsample' in subname3) and ('layer3' in subname or 'layer4' in subname):
                                    setattr(submodule3, subname4,nn.Sequential(submodule4, self.dropout))

        for name, module in self.base_model.named_modules():
            if name == '':
                for subname, submodule in module.named_children():
                    for subname2, submodule2 in submodule.named_children():
                        for subname3, submodule3 in submodule2.named_children():
                            if ('relu' in subname3 or 'bn2' in subname3) and ('layer3' in subname or 'layer4' in subname):
                                setattr(submodule2, subname3,nn.Sequential(submodule3, self.dropout))
                            
        
    def forward(self, x):
        x = self.base_model(x)
        return x
    
    
    @torch.no_grad()
    def get_prediction_stats(self, x: torch.Tensor, n_samples, batch_size) -> torch.Tensor:
        """Calculate the mean prediction across all MC drops."""
        
        
        def forward_impl(self, x):
            # See note [TorchScript super()]
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            #x = self.layer3(x)
            #x = self.layer4(x)

            #x = self.avgpool(x)
            #x = torch.flatten(x, 1)
            #x = self.fc(x)

            return x
        
        def from_avg(self,x):
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            
            return x

        #up_model = nn.Sequential(*list(list(self.children())[0].children()))
        up_model = nn.Sequential(*list(self.base_model.children()))
        down_model = nn.Sequential(*list(list(self.children())[0].children())[-2:])
        
        preds = torch.zeros((batch_size, self.num_classes, n_samples))
        
        #model = self.base_model
        #delattr(model,'fc')
        
        x_base = forward_impl(self.base_model, x)
        #print(x_base.size())
        #print(self.base_model)
        
        for i in range(n_samples):
            #logits = forward_impl(self.base_model, x)
            logits = from_avg(self.base_model,x_base)
            #logits = self.base_model.fc(x_base)
            #logits = self(x)
            #print(logits.size())
            #probs = torch.softmax(logits, dim=-1)
            probs = logits
            preds[:,:,i] = probs
        
        mean = torch.mean(preds,dim=2)
        std = torch.std(preds,dim=2)
        
        return mean, std, preds
    
    
class UncertaintyResNetPretrainedSoftmax(nn.Module):
    def __init__(self, num_classes: int, dropout_rate: float = 0.25, pretrained_weights = True):
        super().__init__()
        
        self.num_classes = num_classes
        
        if pretrained_weights:
            model_weights = ResNet18_Weights.DEFAULT
            self.base_model = resnet18(weights=model_weights)
            self.base_model.fc = torch.nn.Linear(in_features=self.base_model.fc.in_features,out_features=num_classes)
        else:
            self.base_model = resnet18()
            self.base_model.fc = torch.nn.Linear(in_features=self.base_model.fc.in_features,out_features=num_classes)
        
        w = self.base_model.conv1.weight
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=2, bias=False)
        self.base_model.conv1.weight = nn.Parameter(torch.mean(w, dim=1, keepdim=True))
        
        #net = net.to(dev)
        
        # Adding dropout layers inside residual blocks
        #self.dropouts = nn.ModuleList([nn.Dropout3d(p=dropout_rate) for _ in range(17)])
        self.dropout = nn.Dropout(p=dropout_rate)
        
        for name, module in self.base_model.named_modules():
            if name == '':
                #print(name)
                for subname, submodule in module.named_children():
                    #print(subname)
                    for subname2, submodule2 in submodule.named_children():
                        for subname3, submodule3 in submodule2.named_children():
                            for subname4, submodule4 in submodule3.named_children():
                                if '1' in subname4 and 'downsample' in subname3:
                                    setattr(submodule3, subname4,nn.Sequential(submodule4, self.dropout))

        for name, module in self.base_model.named_modules():
            if name == '':
                for subname, submodule in module.named_children():
                    for subname2, submodule2 in submodule.named_children():
                        for subname3, submodule3 in submodule2.named_children():
                            if 'relu' in subname3 or 'bn2' in subname3:
                                setattr(submodule2, subname3,nn.Sequential(submodule3, self.dropout))
                            
        for name, module in self.base_model.named_modules():
            #print(f"name -{name}-")
            if name == '':
                for subname, submodule in module.named_children():
                    #print(f" subname: {subname}")
                    if 'relu' in subname:
                        setattr(module, subname,nn.Sequential(submodule, self.dropout))

#         # List of modules to modify
#         modules_to_modify = []
        
#         # Add dropout after the ReLU layers
#         for name, module in self.base_model.named_modules():
#             if 'relu' in name or 'bn2' in name or 'downsample.1' in name:
#                 modules_to_modify.append((name, module))

# #         # Perform modifications outside the loop
#         for name, module in modules_to_modify:
#             setattr(self.base_model, name, nn.Sequential(module, self.dropout))
        
        #self.n_samples = n_samples
        #print(self.base_model.named_modules())
        
    def forward(self, x):
        x = torch.softmax(self.base_model(x),dim=-1)
        return x
    
    @torch.no_grad()
    def get_prediction_stats(self, x: torch.Tensor, n_samples, batch_size) -> torch.Tensor:
        """Calculate the mean prediction across all MC drops."""
        
        preds = torch.zeros((batch_size, self.num_classes, n_samples))
            
        for i in range(n_samples):
            logits = self(x)
            #probs = torch.softmax(logits, dim=-1)
            probs = logits
            preds[:,:,i] = probs
        
        mean = torch.mean(preds,dim=2)
        std = torch.std(preds,dim=2)
        
        return mean, std, preds