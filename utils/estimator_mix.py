work_path = "/dss/dsshome1/lxc0B/di29let/ondemand/Projects/Repositories/certainty-estimator"
import torch
import os
import sys
import numpy as np
sys.path.append(work_path)
from sklearn.linear_model import LogisticRegression
from scipy.interpolate import CubicSpline


class LogisticRegressionMix():
    """ 
    A class that fits multiple Logistic Regression masking level-specific classifiers 
    and supports interpolation of predictions based on mask levels.
    """

    def __init__(self, C, mask_levels):
        """ 
        Initialize the LogisticRegressionMix class with the regularization 
        parameter 'C' and mask levels.
        """
        self.mask_levels = mask_levels
        self.C = C
        
    def fit(self, X, y):
        """
        Fit logistic regression models for each training sample.
        
        Parameters:
        X : ndarray
            The input features (3D array). First dimension refers to masking levels.
        y : ndarray
            The target labels (2D array). First dimension refers to masking levels.
        """
        self.y = y
        self.mixclf = []
        for i in range(X.size(0)):
            Xtrain = X[i,:,:].cpu() # Extract training data for each masking level
            ytrain = y[i,:].squeeze() # Extract corresponding labels
            
            # Train a logistic regression model
            self.mixclf.append(LogisticRegression(C=self.C,random_state=0,max_iter=3000).fit(Xtrain, ytrain))
        return self
    
    def predict_proba(self, Xtest, mask_levels=None):
        """
        Predict probabilities using the fitted logistic regression models.
        
        Parameters:
        X_test : ndarray
            The input features for testing.
        mask_levels : list or ndarray, optional
            Levels at which to interpolate the predictions (default is None).
        
        Returns:
        ypred : ndarray
            Predicted probabilities after interpolation.
        """
        def direct_clf(X):
            """
            Directly predict probabilities for each instance assuming 
            the standard masking levels.
            
            Parameters:
            X : ndarray
                Input features.
            
            Returns:
            y : ndarray
                Predicted probabilities for each sample.
            """
            y = np.zeros((X.shape[0],X.shape[1]))
            for i, clf in enumerate(self.mixclf):
                #print(clf)
                y[i,:] = clf.predict_proba(X[i,:,:])[:,1]
            return y
        
        def interp_clf(X):
            """
            Predict probabilities for a single instance to interpolate later.
            
            Parameters:
            X : ndarray
                Input features for a single instance.
            
            Returns:
            y : ndarray
                Predicted probabilities for interpolation.
            """
            if X.dim()<3:
                X = X.unsqueeze(0)
            
            y = np.zeros((len(self.mixclf),X.shape[1]))
            
            for i, clf in enumerate(self.mixclf):
                #print(clf)
                y[i,:] = clf.predict_proba(X[0,:,:])[:,1]
            return y

        def confidence_interpolate(y, level):
            """
            Interpolate predicted probabilities based on confidence level.
            
            Parameters:
            y : ndarray
                Predicted probabilities from different classifiers.
            level : float
                Mask level for interpolation.
            
            Returns:
            float
                Interpolated probability for the given mask level.
            """
            cs = CubicSpline(self.mask_levels,y)
            return cs(level)
        
        # If no mask levels are provided, perform direct classification
        if mask_levels is None:
            ypred = direct_clf(Xtest)
        else:    
            yall = interp_clf(Xtest)
            
            # Ensure mask levels is a list or ndarray
            if not isinstance(mask_levels, (list, np.ndarray)):
                mask_levels = [mask_levels]
            
            # Array to hold interpolated predictions
            yinterp = np.zeros((len(mask_levels),yall.shape[1]))
            
            # Interpolate predictions for each instance
            for i,level in enumerate(mask_levels):
                for p in range(yall.shape[1]):
                    yinterp[i,p] = confidence_interpolate(yall[:,p],level)
            ypred = yinterp
        
        return ypred


def train_estimator_mix(masking_levels, drop_names, C=[1,1],dropout_predictions=False,features_file = 'uncertainty_all.pth',
                        clf=None, estimator_type = ['dropout','hfeatures']):
    
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
            Xtrain = fts
            #print(Xtrain.shape,wrong.shape)
            #gind = np.apply_along_axis(lambda x: np.where(np.isnan(x)==0)[0], 0, Xtrain[:,:,1])
            #print(gind.shape)
            #Xtrain = Xtrain[gind[:,0],gind[:,1]]
            wrong_train = wrong
            y = np.apply_along_axis(lambda x: np.where(x==0,1,0),0,wrong_train)
            #print(y.shape)
            #fit classifier
            if not clf:
                uncertainty_drops[type][drop_name]['clf'] = LogisticRegressionMix(C=C[i],mask_levels=masking_levels).fit(Xtrain, y)
            else:
                uncertainty_drops[type][drop_name]['clf'] = clf[type][drop_name]['clf']
    
    return uncertainty_drops