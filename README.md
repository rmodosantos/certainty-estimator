# Certainty estimator: exploring ways to estimate neural network classifier prediction certainty

## Motivation
Neural networks have achieved human-like performance in numerous image classification tasks. However, interpreting the outputs of these "black box" models remains a challenge. This issue is especially critical in the context of medical imaging, where understanding the rationale behind a model's decision is crucial. The ability for an AI model to convey its prediction certainty is highly desirable, as it helps identify ambiguous images that may require further examination by a healthcare professional.

<div style="text-align: center;">
<img src="Uncertainty_DALLE3.png" alt="Show uncertainty" width="600" />
</div>

In this project I explore ways to implement certainty estimation on the predictions of a commonly used covolutional neural network architecture.