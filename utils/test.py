
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


estimator_type = ['dropout','hfeatures']


def train_estimators(drop_names,C=[1,1], dropout_predictions=False,features_file = 'uncertainty_all.pth',clf=None):
    uncertainty_drops = {}
    
    
    for type in estimator_type:
        uncertainty_drops[type] = {key: {} for key in drop_names}
    
    for drop_name in drop_names:
        
        model_path = f"/dss/dssfs02/lwp-dss-0001/pr84qa/pr84qa-dss-0000/ricardo/data/Projects/MRI_classification/models/resnet18/pretrained/dropout_allconvs/dp_rate_{drop_name}_nonorm_masked_extendedtrain06/29_05_2024"     
        uncertainty_tm = torch.load(os.path.join(model_path,features_file))
        
        for i, type in enumerate(estimator_type):
        
            print(drop_name)
            
            for key in ['train','validation']:
                uncertainty_drops[type][drop_name][key] = {}
                for data_name in ['features','wrong']:
                    uncertainty_drops[type][drop_name][key][data_name] = uncertainty_tm[key][data_name][type]
            
            #fit logistic model
            if dropout_predictions:
                pred_type = 'dropout'
            else:
                pred_type = type
                
            fts = uncertainty_drops[type][drop_name]['train']['features']
            wrong = uncertainty_drops[pred_type][drop_name]['train']['wrong']
            Xtrain = fts.view(-1,fts.size(2))
            gind = torch.where(torch.isnan(Xtrain[:,1])==0)[0]
            Xtrain = Xtrain[gind,:]
            wrong_train = wrong.view(-1,1)[gind,:]
            y = torch.where(wrong_train==0,1,0).squeeze()

            #fit classifier
            if not clf:
                uncertainty_drops[type][drop_name]['clf'] = LogisticRegression(C=C[i],random_state=0,max_iter=3000).fit(Xtrain, y)
            else:
                uncertainty_drops[type][drop_name]['clf'] = clf[type][drop_name]['clf']
    
    return uncertainty_drops

def get_estimator_accuracy(uncertainty_drops,drop_names,dataset='validation',estimator_mix=False,masking_levels=None):

    hbins = np.arange(0.3,1.02,0.02)
    uncertainty_drops['model confidence'] = hbins[:-1]+0.01
    
    
    for type in estimator_type:
        for drop_name in drop_names:
            
            fts_val = uncertainty_drops[type][drop_name][dataset]['features']
            wrong_val = uncertainty_drops[type][drop_name][dataset]['wrong']
            X = fts_val.view(-1,fts_val.size(2))

            # Histograms of estimator output probability of a sample being correct
            # All predictions
            if estimator_mix:
                pred_proba = uncertainty_drops[type][drop_name]['clf'].predict_proba(fts_val).reshape(-1,1)
            else:
                
                pred_proba = uncertainty_drops[type][drop_name]['clf'].predict_proba(X)[:,1]
                
            b = np.histogram(pred_proba,bins=hbins)
            
            # Wrong predictions 
            c = np.histogram(pred_proba[wrong_val.view(-1,1).squeeze()==1],bins=hbins)
            
            #get global nan indices
            with np.errstate(divide='ignore', invalid='ignore'):
                ind_nan = ~np.isnan(1-c[0]/b[0])
                uncertainty_drops[type][drop_name]['global_nan_bin'] = ind_nan
               
            #store stats in dictionary
            uncertainty_drops[type][drop_name]['acc_total'] = 1-c[0]/b[0]
            uncertainty_drops[type][drop_name]['conf_total'] = b[0]
            uncertainty_drops[type][drop_name]['accuracy'] = 1 - torch.sum(wrong_val.view(-1,1))/wrong_val.view(-1,1).size(0)

            # The same for each masking extent
            predm = []
            pred_acc = []
            uncertainty_drops[type][drop_name]['nan_bin'] = []

            for ind in range(fts_val.size(0)):
                if estimator_mix:
                    pred_proba = uncertainty_drops[type][drop_name]['clf'].predict_proba(fts_val[ind,:,:].squeeze(),
                                                                                         mask_levels=masking_levels[ind]).reshape(-1,1)
                else:
                    pred_proba = uncertainty_drops[type][drop_name]['clf'].predict_proba(fts_val[ind,:,:].squeeze())[:,1]
                    
                bm,_ = np.histogram(pred_proba,bins=hbins)
                cm,_ = np.histogram(pred_proba[wrong_val[ind,:]==1],bins=hbins)

                with np.errstate(divide='ignore', invalid='ignore'):
                    pred_acc.append(1-cm/bm)
                    predm.append(bm/np.max(bm))
                    ind_nan = ~np.isnan(1-cm/bm)
                    uncertainty_drops[type][drop_name]['nan_bin'].append(ind_nan)

            uncertainty_drops[type][drop_name]['acc_bin'] = np.array(pred_acc)
            uncertainty_drops[type][drop_name]['conf_bin'] = np.array(predm)
            uncertainty_drops[type][drop_name]['nan_bin'] = np.array(uncertainty_drops[type][drop_name]['nan_bin'])
    
    return uncertainty_drops


def get_estimator_stats(uncertainty_drops,corruption_levels,drop_names):
    
    stat_names = ['accuracy','hconf']
    stats = {}
    for type in estimator_type:
        stats[type] = {key:{} for key in ['total', 'binned']}
        for key in stats[type]:
            stats[type][key] = {k:[] for k in stat_names}

    for type in estimator_type:
        for drop_name in drop_names:
            #total
            ind_nan = uncertainty_drops[type][drop_name]['global_nan_bin']
            xp = uncertainty_drops['model confidence'].reshape(-1,1)[ind_nan]
            yp = uncertainty_drops[type][drop_name]['acc_total'][ind_nan]
            stats[type]['total']['accuracy'].append(uncertainty_drops[type][drop_name]['accuracy'])

            #binned
            h_conf = []
            
            for i in range(corruption_levels.shape[0]):
                ind_nan = uncertainty_drops[type][drop_name]['nan_bin'][i,:]
                yp = uncertainty_drops[type][drop_name]['acc_bin'][i,ind_nan]
                xp = uncertainty_drops['model confidence'][ind_nan].reshape(-1,1)
                h_conf.append(np.mean(yp[xp[:,0]>0.9]))

            stats[type]['binned']['hconf'].append(h_conf)

        stats[type]['binned']['hconf'] = np.array(stats[type]['binned']['hconf']).squeeze()
    
    return stats
