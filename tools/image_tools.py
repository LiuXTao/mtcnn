import numpy as np
import torch
from torchvision.transforms import transforms

transform = transforms.ToTensor

def convert_image_to_tensor(image):
    '''
           Parameters:
        ----------
        image: numpy array , h * w * c

        Returns:
        -------
        image_tensor: pytorch.FloatTensor, c * h * w

    '''
    image = image.astype(np.float32)

    return transform(image)


def convert_chwTensor_to_hwcNumpy(tensor):
    """convert a group images pytorch tensor(count * c * h * w) to numpy array images(count * h * w * c)
            Parameters:
            ----------
            tensor: numpy array , count * c * h * w

            Returns:
            -------
            numpy array images: count * h * w * c
    """
    if isinstance(tensor, torch.FloatTensor):
        return np.transpose(tensor.detach().numpy(), (0, 2, 3, 1))
    else:
        raise Exception(
            "covert b*c*h*w tensor to b*h*w*c numpy error.This tensor must have 4 dimension of float data type.")
