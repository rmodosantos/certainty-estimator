# Certainty estimator: extracting prediction certainty from a neural network image classifier

## Motivation
Neural networks have achieved human-like performance in numerous image classification tasks. However, interpreting the outputs of these "black box" models remains a challenge. This issue is especially critical in the context of medical imaging, where understanding the rationale behind a model's decision is crucial. The ability for an AI model to convey its prediction certainty is highly desirable, as it helps identify ambiguous images that may require further examination by a healthcare professional.

<img src="Uncertainty_DALLE3.png" alt="Show uncertainty" width="500" />
One of the main goals of interpretable AI is do design models that can output the certainty of their predictions.

In this project I explore ways to implement certainty estimation on the predictions of a commonly used covolutional neural network (CNN) architecture. In this context, certainty represents the probability of a given prediction to be correct.

## Model implementation at a glance

I picked a commonly used CNN architecture, the ResNet18, to classify different types of brain tumors in MRI images. Training was performed using dropout in all layers, which served as a regularization method at this stage, as schematized below. 


### Monte Carlo dropout based uncertainty estimation
Monte Carlo (MC) dropout has been proposed as a method to estimate neural network model prediction uncertainty. In this approach, dropout used during training is kept on during inference, enabling the sampling of multiple slightly different CNN outputs. The end-result is a distribution of CNN outputs for each class, insted of the deterministic CNN output consisting of a single output per class. 

Uncertainty, can then, in principle, be extracted from metrics derived from the MC dropout samples, such as its standard deviation. However, it is often not straightforward to derive the most informative of such metrics and translate it into a certainty value. To circunvent this problem, here I trained a logistic regression to predict whether the CNN prediction is correct, having the MC dropoout samples as input, as shematized below. Remarkably, the output of the logistic regression estimator can be given as a probability, corresponding to an estimation of the CNN prediction certainty.