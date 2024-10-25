
work_path = "/dss/dsshome1/lxc0B/di29let/ondemand/Projects/Repositories/certainty-estimator"
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sys
import os
sys.path.append(work_path)
from utils.models import UncertaintyResNetPretrained, UncertaintyResNetPretrainedBNnoTrack, UncertaintyResNetPretrainedFCdrop, UncertaintyResNetPretrainedLayer4drop, UncertaintyResNetPretrainedLayer3drop, UncertaintyResNetPretrainedSoftmax
import numpy as np
from sklearn.linear_model import LogisticRegression

def test(net, data, criterion, dropout_p,dev, batch_size=10, printr=False):
    """Function to test neural network predictions on a given dataset
            Inputs:
                net: Network to test
                sampler: Subset random sampler used to sample validation or test sets
                batch_size: size of batchs to compute predictions on
                printr: Whether to show performance results"""

    testloader = iter(DataLoader(data,batch_size=batch_size))
    
    net.dropout.p = 0
    net.to(dev)
    
    correct = 0
    total = 0
    labels = []
    predicted = []
    losses = []
    with torch.no_grad():
        for _ in range(10):
            
            image,label = next(testloader)
            
            # Compute predictions from the class with max output
            outputs=net(image).cpu()
            prediction = np.argmax(outputs,axis=1)
            print(prediction)
            total += label.size(0)
            correct += (prediction == label.cpu()).sum().item()
            
            labels.append(list(label.cpu()))
            predicted.append(list(prediction))
            loss = criterion(outputs,label.cpu()).item()
            losses.append(loss)
    
    #compute confusion matrix
    cm = confusion_matrix(labels[0],predicted[0])
    
    if printr:
        print(cm)
        print('Accuracy:', 100*correct/total)
    
    net.dropout.p = dropout_p
    
    return cm, 100*correct/total, predicted, labels, losses


def SaveCheckpoint(path,bsize,lr,beta1,beta2,epocs):
    """Save model checkpoint with training parameters.

    Args:
        path (str): Path where to save the checkpoint
        batch_size (int): Size of training batches
        learning_rate (float): Learning rate used in training
        beta1 (float): Beta1 parameter for optimizer
        beta2 (float): Beta2 parameter for optimizer
        epochs (int): Number of training epochs

    Returns:
        None
    """
    
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'batch_size': bsize,
        'learning_rate': lr,
        'beta1': beta1,
        'beta2': beta2,
        'epochs': epocs
    },path)
    

def load_model(path,dev,dropout_rate):
    """
    Load a pre-trained model along with its learning parameters.

    Args:
        path (str): The file path to the saved model checkpoint.
        dev (str): Device where to store the tensors.
        dropout_rate (float): The dropout rate to use in the model.

    Returns:
        net (torch.nn.Module): The loaded neural network model.
        learning_params (dict): A dictionary containing the model's learning parameters such as:
            - learning_rate (float)
            - batch_size (int)
            - beta1 (float)
            - beta2 (float)
            - epochs (int)
    """
    
    # Load the saved model checkpoint from the specified path
    model_dict = torch.load(path,map_location=dev)
    
    # Extract the model state dictionary
    state_dict = model_dict['model_state_dict']
    
    # Initialize the model architecture with the specified dropout rate
    net = UncertaintyResNetPretrained(num_classes=3,dropout_rate=dropout_rate,pretrained_weights=True)
    net.load_state_dict(state_dict,strict=False)
    net.to(dev)
    
    # Extract and return learning parameters
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


estimator_type = ['dropout','hfeatures']


def train_estimators(drop_names,C=[1,1], dropout_predictions=False,features_file = 'uncertainty_all.pth',clf=None):
    """
    Train uncertainty estimators (logistic classifiers) on extracted features for different dropout rates.

    This function loads the feature representations from previously saved models, and uses them to 
    train logistic regression models to predict whether the model's classification is correct or not 
    (based on 'wrong' labels). The trained classifiers are stored in a dictionary for later use.

    Args:
        drop_names (str): dropout names matching name patterns of saved files.
        C (list of float, optional): Regularization strengths for the logistic regression classifiers 
                                     for each estimator type. Default is [1, 1].
        dropout_predictions (bool, optional): If True, train using dropout-based predictions. 
                                              If False, train on hidden features ('hfeatures'). Default is False.
        features_file (str, optional): Filename of the .pth file containing features and wrong labels. 
                                       Default is 'uncertainty_all.pth'.
        clf (dict, optional): Pre-trained classifiers for each dropout rate and estimator type. 
                              If None, new classifiers will be trained. Default is None.

    Returns:
        dict: A dictionary containing trained classifiers for each dropout rate and estimator type.
    """
    
    # Initialize dictionary to store uncertainty estimators
    uncertainty_drops = {}
    for type in estimator_type:
        uncertainty_drops[type] = {key: {} for key in drop_names}
    
    for drop_name in drop_names:
        # Load the features and wrong labels from the saved file
        model_path = f"/dss/dssfs02/lwp-dss-0001/pr84qa/pr84qa-dss-0000/ricardo/data/Projects/MRI_classification/models/resnet18/pretrained/dropout_allconvs/dp_rate_{drop_name}_nonorm_masked_extendedtrain06/29_05_2024"     
        uncertainty_tm = torch.load(os.path.join(model_path,features_file))
        
        for i, type in enumerate(estimator_type):
        
            print(drop_name)
            
            for key in ['train','validation']:
                uncertainty_drops[type][drop_name][key] = {}
                for data_name in ['features','wrong']:
                    uncertainty_drops[type][drop_name][key][data_name] = uncertainty_tm[key][data_name][type]
            
            if dropout_predictions:
                pred_type = 'dropout'
            else:
                pred_type = type

            # Prepare the training features and labels for the logistic regression model
            fts = uncertainty_drops[type][drop_name]['train']['features']        
            wrong = uncertainty_drops[pred_type][drop_name]['train']['wrong']
            
            # Reshape the features and remove any NaN values
            Xtrain = fts.view(-1,fts.size(2))
            gind = torch.where(torch.isnan(Xtrain[:,1])==0)[0]
            Xtrain = Xtrain[gind,:]
            wrong_train = wrong.view(-1,1)[gind,:]

            # Create binary labels
            y = torch.where(wrong_train==0,1,0).squeeze()

            # Train the logistic regression classifier
            if not clf:
                # Train a new logistic regression model
                uncertainty_drops[type][drop_name]['clf'] = LogisticRegression(C=C[i],random_state=0,max_iter=3000).fit(Xtrain, y)
            else:
                # Use the provided pre-trained classifier
                uncertainty_drops[type][drop_name]['clf'] = clf[type][drop_name]['clf']
    
    return uncertainty_drops
    

def get_estimator_accuracy(uncertainty_drops,drop_names,dataset='validation',estimator_mix=False,masking_levels=None):
    """
    Computes and updates the accuracy and confidence statistics for different uncertainty estimators.

    This function calculates accuracy, confidence (certainty), and various metrics across a range of certainty 
    estimators (dropout or hfeatures). It computes accuracy metrics for the entire dataset as well as for different 
    masking levels.

    Args:
        uncertainty_drops (dict): A dictionary containing trained certainty estimators, their predictions,
                                  and 'wrong' labels (whether the prediction was wrong).
        drop_names (list of str): List of dropout rates (as strings) to evaluate.
        dataset (str, optional): The dataset to compute accuracy for, either 'train' or 'validation'. 
                                 Default is 'validation'.
        estimator_mix (bool, optional): If True, combines different estimators; if False, uses individual estimators.
                                        Default is False.
        masking_levels (list, optional): List of masking levels, used if estimator_mix is True. Default is None.

    Returns:
        dict: Updated `uncertainty_drops` dictionary with computed accuracy and confidence statistics.
    """
    
    # Define histogram bins for confidence (certainty) values
    hbins = np.arange(0.3,1.02,0.02)
    uncertainty_drops['model confidence'] = hbins[:-1]+0.01
    
    
    for type in estimator_type:
        for drop_name in drop_names:
            # Extract validation features and wrong predictions
            fts_val = uncertainty_drops[type][drop_name][dataset]['features']
            wrong_val = uncertainty_drops[type][drop_name][dataset]['wrong']
            X = fts_val.view(-1,fts_val.size(2))

            # Compute the predicted probabilities of correct classifications (certainty)
            if estimator_mix:
                pred_proba = uncertainty_drops[type][drop_name]['clf'].predict_proba(fts_val).reshape(-1,1)
            else:
                pred_proba = uncertainty_drops[type][drop_name]['clf'].predict_proba(X)[:,1]
                
            # Create histograms for all predictions
            b = np.histogram(pred_proba,bins=hbins)
            
            # Create histograms for wrong predictions
            c = np.histogram(pred_proba[wrong_val.view(-1,1).squeeze()==1],bins=hbins)
            
            # Identify bins where we don't have NaN values (to avoid dividing by zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                ind_nan = ~np.isnan(1-c[0]/b[0])
                uncertainty_drops[type][drop_name]['global_nan_bin'] = ind_nan
               
            # Store overall accuracy and confidence
            uncertainty_drops[type][drop_name]['acc_total'] = 1-c[0]/b[0]
            uncertainty_drops[type][drop_name]['conf_total'] = b[0]
            uncertainty_drops[type][drop_name]['accuracy'] = 1 - torch.sum(wrong_val.view(-1,1))/wrong_val.view(-1,1).size(0)

            # Initialize lists to store per-mask accuracy and confidence
            predm = []
            pred_acc = []
            uncertainty_drops[type][drop_name]['nan_bin'] = []
            
            # Compute accuracy and certainty for each masking level
            for ind in range(fts_val.size(0)):
                if estimator_mix:
                    pred_proba = uncertainty_drops[type][drop_name]['clf'].predict_proba(fts_val[ind,:,:].squeeze(),
                                                                                         mask_levels=masking_levels[ind]).reshape(-1,1)
                else:
                    pred_proba = uncertainty_drops[type][drop_name]['clf'].predict_proba(fts_val[ind,:,:].squeeze())[:,1]
                
                # Compute per-mask histograms
                bm,_ = np.histogram(pred_proba,bins=hbins)
                cm,_ = np.histogram(pred_proba[wrong_val[ind,:]==1],bins=hbins)
                
                # Calculate per-mask accuracy and certainty
                with np.errstate(divide='ignore', invalid='ignore'):
                    pred_acc.append(1-cm/bm)
                    predm.append(bm/np.max(bm))
                    ind_nan = ~np.isnan(1-cm/bm)
                    uncertainty_drops[type][drop_name]['nan_bin'].append(ind_nan)
            
            # Store per-mask accuracy and confidence
            uncertainty_drops[type][drop_name]['acc_bin'] = np.array(pred_acc)
            uncertainty_drops[type][drop_name]['conf_bin'] = np.array(predm)
            uncertainty_drops[type][drop_name]['nan_bin'] = np.array(uncertainty_drops[type][drop_name]['nan_bin'])
    
    return uncertainty_drops

