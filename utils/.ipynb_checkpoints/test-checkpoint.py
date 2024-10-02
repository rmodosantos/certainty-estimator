
work_path = "/dss/dsshome1/lxc0B/di29let/ondemand/Projects/Repositories/certainty-estimator"
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sys
sys.path.append(work_path)
from utils.models import UncertaintyResNetPretrained, UncertaintyResNetPretrainedBNnoTrack, UncertaintyResNetPretrainedFCdrop, UncertaintyResNetPretrainedLayer4drop, UncertaintyResNetPretrainedLayer3drop, UncertaintyResNetPretrainedSoftmax
import numpy as np

def test(net, data, criterion, dropout_p,dev, batch_size=10, printr=False):
    """Function to test neural network predictions on a given dataset
            Inputs:
                net: Network to test
                sampler: Subset random sampler used to sample validation or test sets
                batch_size: size of batchs to compute predictions on
                printr: Whether to show performance results"""

    testloader = iter(DataLoader(data,batch_size=batch_size))
    #net = net.float()
    net.dropout.p = 0
    net.to(dev)
    #net.eval()
    
    correct = 0
    total = 0
    labels = []
    predicted = []
    losses = []
    with torch.no_grad():
        for _ in range(10):
            
            image,label = next(testloader)
            #compute predictions from the class with max output
            outputs=net(image).cpu()
            prediction = np.argmax(outputs,axis=1)
            print(prediction)
            total += label.size(0)
            correct += (prediction == label.cpu()).sum().item()
            
            labels.append(list(label.cpu()))
            predicted.append(list(prediction))
            loss = criterion(outputs,label.cpu()).item()
            losses.append(loss)
    
    #print(total)
    #compute confusion matrix
    cm = confusion_matrix(labels[0],predicted[0])
    
    if printr:
        print(cm)
        print('Accuracy:', 100*correct/total)
    
    net.dropout.p = dropout_p
    
    return cm, 100*correct/total, predicted, labels, losses


def SaveCheckpoint(path,bsize,lr,beta1,beta2,epocs):
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'batch_size': bsize,
        'learning_rate': lr,
        'beta1': beta1,
        'beta2': beta2,
        'epochs': epocs
    },path)
    

def load_model(path,dev, dropout_rate):
    model_dict = torch.load(path,map_location=dev)
#     model_dict['model_state_dict'] = {k.replace('base_model.',''):v
#                                    for k,v in model_dict['model_state_dict'].items()}
    state_dict = model_dict['model_state_dict']
    
    #print(state_dict.keys())
    #net = resnet18(spatial_dims=2, n_input_channels=1, num_classes=3)
    #net = resnet18()
    #net.fc = torch.nn.Linear(in_features=net.fc.in_features,out_features=3)
    net = UncertaintyResNetPretrained(num_classes=3,dropout_rate=dropout_rate,pretrained_weights=True)
    net.load_state_dict(state_dict,strict=False)
    net.to(dev)
    learning_params = {key: model_dict[key] for key in ['learning_rate','batch_size','beta1','beta2','epochs']}
    
    return net, learning_params


def load_model_noTrack(path,dev, dropout_rate):
    model_dict = torch.load(path,map_location=dev)
#     model_dict['model_state_dict'] = {k.replace('base_model.',''):v
#                                    for k,v in model_dict['model_state_dict'].items()}
    state_dict = model_dict['model_state_dict']
    
    #print(state_dict.keys())
    #net = resnet18(spatial_dims=2, n_input_channels=1, num_classes=3)
    #net = resnet18()
    #net.fc = torch.nn.Linear(in_features=net.fc.in_features,out_features=3)
    net = UncertaintyResNetPretrainedBNnoTrack(num_classes=3,dropout_rate=dropout_rate,pretrained_weights=True)
    net.load_state_dict(state_dict,strict=False)
    net.to(dev)
    learning_params = {key: model_dict[key] for key in ['learning_rate','batch_size','beta1','beta2','epochs']}
    
    return net, learning_params


def load_model_FCdrop(path,dev, dropout_rate):
    model_dict = torch.load(path,map_location=dev)
#     model_dict['model_state_dict'] = {k.replace('base_model.',''):v
#                                    for k,v in model_dict['model_state_dict'].items()}
    state_dict = model_dict['model_state_dict']
    
    #print(state_dict.keys())
    #net = resnet18(spatial_dims=2, n_input_channels=1, num_classes=3)
    #net = resnet18()
    #net.fc = torch.nn.Linear(in_features=net.fc.in_features,out_features=3)
    net = UncertaintyResNetPretrainedFCdrop(num_classes=3,dropout_rate=dropout_rate,pretrained_weights=True)
    net.load_state_dict(state_dict,strict=False)
    net.to(dev)
    learning_params = {key: model_dict[key] for key in ['learning_rate','batch_size','beta1','beta2','epochs']}
    
    return net, learning_params


def load_model_layer4drop(path,dev, dropout_rate):
    model_dict = torch.load(path,map_location=dev)
#     model_dict['model_state_dict'] = {k.replace('base_model.',''):v
#                                    for k,v in model_dict['model_state_dict'].items()}
    state_dict = model_dict['model_state_dict']
    
    #print(state_dict.keys())
    #net = resnet18(spatial_dims=2, n_input_channels=1, num_classes=3)
    #net = resnet18()
    #net.fc = torch.nn.Linear(in_features=net.fc.in_features,out_features=3)
    net = UncertaintyResNetPretrainedLayer4drop(num_classes=3,dropout_rate=dropout_rate,pretrained_weights=True)
    net.load_state_dict(state_dict,strict=False)
    net.to(dev)
    learning_params = {key: model_dict[key] for key in ['learning_rate','batch_size','beta1','beta2','epochs']}
    
    return net, learning_params


def load_model_layer3drop(path,dev, dropout_rate):
    model_dict = torch.load(path,map_location=dev)
#     model_dict['model_state_dict'] = {k.replace('base_model.',''):v
#                                    for k,v in model_dict['model_state_dict'].items()}
    state_dict = model_dict['model_state_dict']
    
    #print(state_dict.keys())
    #net = resnet18(spatial_dims=2, n_input_channels=1, num_classes=3)
    #net = resnet18()
    #net.fc = torch.nn.Linear(in_features=net.fc.in_features,out_features=3)
    net = UncertaintyResNetPretrainedLayer3drop(num_classes=3,dropout_rate=dropout_rate,pretrained_weights=True)
    net.load_state_dict(state_dict,strict=False)
    net.to(dev)
    learning_params = {key: model_dict[key] for key in ['learning_rate','batch_size','beta1','beta2','epochs']}
    
    return net, learning_params


def load_model_softmax(path,dev, dropout_rate):
    model_dict = torch.load(path,map_location=dev)
#     model_dict['model_state_dict'] = {k.replace('base_model.',''):v
#                                    for k,v in model_dict['model_state_dict'].items()}
    state_dict = model_dict['model_state_dict']
    
    #print(state_dict.keys())
    #net = resnet18(spatial_dims=2, n_input_channels=1, num_classes=3)
    #net = resnet18()
    #net.fc = torch.nn.Linear(in_features=net.fc.in_features,out_features=3)
    net = UncertaintyResNetPretrainedSoftmax(num_classes=3,dropout_rate=dropout_rate,pretrained_weights=True)
    net.load_state_dict(state_dict,strict=False)
    net.to(dev)
    learning_params = {key: model_dict[key] for key in ['learning_rate','batch_size','beta1','beta2','epochs']}
    
    return net, learning_params
