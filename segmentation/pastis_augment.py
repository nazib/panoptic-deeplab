import torch
import torchvision.transforms as T

def applyTransforms(image):

    equalizer = T.RandomEqualize()
    image = equalizer(image.type(torch.uint8))
    ## Random rotation
    rotater = T.RandomRotation(degrees=(0, 180))
    image = rotater(image)

    #Applying color Jitter
    jitter = T.ColorJitter(brightness=.5, hue=.3)
    image = jitter(image)

    ## Gaussian Blurr
    blurrer = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
    image = blurrer(image)

    ## Elastic transform
    #elastic_transformer = T.ElasticTransform(alpha=250.0)
    #image = elastisc_transformer(image)

    autocontraster = T.RandomAutocontrast()
    image =  autocontraster(image)

    return image

