
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import slowfast.datasets.utils as data_utils
from slowfast.visualization.utils import get_layer
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, ToTensor
from typing import List, Dict
import math

def reshape_transform(tensor, height=7, width=7):
    print(tensor.shape)
    result = tensor[:, 40:, :].reshape(tensor.size(0),
    8,height, width, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.permute(0,4,1,2,3)
    #print(result.shape)
    #os.exit()
    return result
def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
class GradCAM:
    """

    """

    def __init__(
        self, model, target_layers, data_mean=0.0, data_std=1.0, colormap="viridis"
    ):
        """
        Args:
            model (model): the model to be used.
            target_layers (list of str(s)): name of convolutional layer to be used to get
                gradients and feature maps from for creating localization maps.
            data_mean (tensor or list): mean value to add to input videos.
            data_std (tensor or list): std to multiply for input videos.
            colormap (Optional[str]): matplotlib colormap used to create heatmap.
                
        """

        self.model = model
        # Run in eval mode.
        self.model.eval()
        self.target_layers = target_layers

        self.gradients = {}
        self.activations = {}
        self.colormap = plt.get_cmap(colormap)
        self.data_mean = data_mean
        self.data_std = data_std
        self._register_hooks()

    def _register_single_hook(self, layer_name):
        """
        Register forward and backward hook to a layer, given layer_name,
        to obtain gradients and activations.
        Args:
            layer_name (str): name of the layer.
        """

        def get_gradients(module, grad_input, grad_output):
            self.gradients[layer_name] = grad_output[0].detach()

        def get_activations(module, input, output):
            self.activations[layer_name] = output.clone().detach()
        #print(layer_name)
        target_layer = self.model.blocks[-1].norm1#get_layer(self.model, layer_name=layer_name)
        target_layer.register_forward_hook(get_activations)
        target_layer.register_backward_hook(get_gradients)

    def _register_hooks(self):
        """
        Register hooks to layers in `self.target_layers`.
        """
        for layer_name in self.target_layers:
            self._register_single_hook(layer_name=layer_name)

    def _calculate_localization_map(self, inputs, labels=None):
        """
        Calculate localization map for all inputs with Grad-CAM.
        Args:
            inputs (list of tensor(s)): the input clips.
            labels (Optional[tensor]): labels of the current input clips.
        Returns:
            localization_maps (list of ndarray(s)): the localization map for
                each corresponding input.
            preds (tensor): shape (n_instances, n_class). Model predictions for `inputs`.
        """
        assert len(inputs) == len(
            self.target_layers
        ), "Must register the same number of target layers as the number of input pathways."
        input_clone = [[inputs]]#[inp.clone() for inp in inputs]
        preds = self.model(input_clone)

        if labels is None:
            score = torch.max(preds, dim=-1)[0]
        else:
            if labels.ndim == 1:
                labels = labels.unsqueeze(-1)
            score = torch.gather(preds, dim=1, index=labels)

        self.model.zero_grad()
        score = torch.sum(score)
        score.backward()
        localization_maps = []
        for i, inp in enumerate([inputs]):
            _, _, T, H, W = inp.size()

            gradients = self.gradients[self.target_layers[i]]
            gradients = reshape_transform(gradients)
            activations = self.activations[self.target_layers[i]]
            activations = reshape_transform(activations)
            B, C, Tg, _, _ = gradients.size()

            weights = torch.mean(gradients.view(B, C, Tg, -1), dim=3)

            weights = weights.view(B, C, Tg, 1, 1)
            localization_map = torch.sum(
                weights * activations, dim=1, keepdim=True
            )
            localization_map[localization_map<0]=0.0*localization_map[localization_map<0]
            localization_map = F.relu(localization_map)
            localization_map = F.interpolate(
                localization_map,
                size=(T, H, W),
                mode="trilinear",
                align_corners=False,
            )
            localization_map_min, localization_map_max = (
                torch.min(localization_map.view(B, -1), dim=-1, keepdim=True)[
                    0
                ],
                torch.max(localization_map.view(B, -1), dim=-1, keepdim=True)[
                    0
                ],
            )
            localization_map_min = torch.reshape(
                localization_map_min, shape=(B, 1, 1, 1, 1)
            )
            localization_map_max = torch.reshape(
                localization_map_max, shape=(B, 1, 1, 1, 1)
            )
            # Normalize the localization map.
            localization_map = (localization_map - localization_map_min) / (
                localization_map_max - localization_map_min + 1e-6
            )
            localization_map = localization_map.data
            #print(localization_map)
            localization_maps.append(localization_map)

        return localization_maps, preds

    def __call__(self, inputs, labels=None, alpha=0.5):
        """
        Visualize the localization maps on their corresponding inputs as heatmap,
        using Grad-CAM.
        Args:
            inputs (list of tensor(s)): the input clips.
            labels (Optional[tensor]): labels of the current input clips.
            alpha (float): transparency level of the heatmap, in the range [0, 1].
        Returns:
            result_ls (list of tensor(s)): the visualized inputs.
            preds (tensor): shape (n_instances, n_class). Model predictions for `inputs`.
        """
        result_ls = []
        localization_maps, preds = self._calculate_localization_map(
            inputs, labels=labels
        )
        for i, localization_map in enumerate(localization_maps):
            # Convert (B, 1, T, H, W) to (B, T, H, W)
            localization_map = localization_map.squeeze(dim=1)
            if localization_map.device != torch.device("cpu"):
                localization_map = localization_map.cpu()
            #heatmap = self.colormap(localization_map*255)
            #heatmap = heatmap[:, :, :, :, :3]
            inputs = [inputs]
            # Permute input from (B, C, T, H, W) to (B, T, H, W, C)
            curr_inp = inputs[i][:,:3].permute(0, 2, 3, 4, 1)
            T = curr_inp.shape[1]
            video = []
            for i in range(T):
                rendered_image = show_cam_on_image(curr_inp[:,i,:,:,:].squeeze().cpu().numpy(), localization_map[:,i,:,:].squeeze().unsqueeze(-1).cpu().numpy())
                video.append(rendered_image)
            video = torch.Tensor(np.stack(video, axis=1))
            #if curr_inp.device != torch.device("cpu"):
            #    curr_inp = curr_inp.cpu()
            #curr_inp = data_utils.revert_tensor_normalize(
            #    curr_inp, self.data_mean, self.data_std
            #)
            #heatmap = torch.from_numpy(heatmap)
            #print(heatmap)
            #curr_inp = alpha * heatmap + (1 - alpha) * curr_inp*255
            # Permute inp to (B, T, C, H, W)
            #curr_inp = curr_inp.permute(0, 1, 4, 2, 3)
            result_ls.append(video)

        return result_ls, preds
