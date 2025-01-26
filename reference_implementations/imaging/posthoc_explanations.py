from captum.attr import LayerGradCam, LayerAttribution,ShapleyValues,IntegratedGradients,NoiseTunnel
import torch

from omnixai.explainers.vision import IntegratedGradientImage,ShapImage,LimeImage
#source: captum.ai
# ImageClassifier takes a single input tensor of images Nx3x32x32,
# and returns an Nx10 tensor of class probabilities.
# It contains a layer conv4, which is an instance of nn.conv2d,
# and the output of this layer has dimensions Nx50x8x8.
 # It is the last convolution layer, which is the recommended
 # use case for GradCAM.
def gradcam(model,input,pred):
    # Computes layer GradCAM for the pred class
    # attribution size matches layer output except for dimension
    layer_gc = LayerGradCam(model, model.conv4)
    attr = layer_gc.attribute(input, pred)
    upsampled_attr = LayerAttribution.interpolate(attr, (124, 124)) # projet back to original image size
    return upsampled_attr


def smoothgrad(model,input,pred):
    # Computes layer smoothgrad for the pred class
    
    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)
    attr = nt.attribute(input, nt_type='smoothgrad',nt_samples=10, target=pred)
    print("ssss",attr.shape)
    # upsampled_attr = LayerAttribution.interpolate(attr, (124, 124)) # projet back to original image size
    return attr

def shapely(model,input,pred):
    # Computes layer smoothgrad for the pred class
    
    sv = ShapleyValues(model)
    feature_mask = torch.zeros((1, 224, 224), dtype=torch.int)

    group_id = 0
    for i in range(0, 224, 4):
        for j in range(0, 224, 4):
            feature_mask[0, i:i+4, j:j+4] = group_id
            group_id += 1
    attr = sv.attribute(input, target=pred, feature_mask=feature_mask)
    print("ssss",attr.shape)
    # upsampled_attr = LayerAttribution.interpolate(attr, (124, 124)) # projet back to original image size
    return attr


def lime(model,input,pred):
    explainer = LimeImage(predict_function=model)
    # Explain the top labels
    explanations = explainer.explain(input, hide_color=0, num_samples=1000)
    explanations.ipython_plot(index=0, class_names=pred)

    
def IG(model,input,pred):  #integrated_gradients
    explainer = IntegratedGradientImage(
    model=model
    )
    # Explain the top labels
    explanations = explainer.explain(input)
    explanations.ipython_plot(index=0, class_names=pred)

def shap(model,input,pred):  #integrated_gradients
    explainer = ShapImage(
    model=model
    )
    # Explain the top labels
    explanations = explainer.explain(input)
    explanations.ipython_plot(index=0)