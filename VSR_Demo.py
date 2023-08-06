import os
import math
import warnings
import numpy as np

import torch
import torch.nn.functional as F

from ModelZoo import load_model
from ModelZoo.utils import Tensor2PIL, PIL2Tensor
from SaliencyModel.VSR_BackProp import GaussianBlurPath
from SaliencyModel.VSR_BackProp import attribution_objective, Path_gradient
from SaliencyModel.VSR_BackProp import saliency_map_PG as saliency_map
from SaliencyModel.attributes import attr_grad
from SaliencyModel.utils import cv2_to_pil, pil_to_cv2
from SaliencyModel.utils import vis_saliency, vis_saliency_kde, click_select_position, grad_abs_norm, prepare_clips, \
    make_pil_grid

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 'BASICVSR@Base' : BASICVSRNet
# 'ICONVSR@Base' : ICONVSRNet
# 'BASICVSRPP@Base' : BASICVSR_Plus_Plus_Net
# 'TTVSR@Base' : TTVSRNet
# 'PSRT@Base' : RethinkingAlignment
model = load_model('BASICVSR@Full')

window_size = 64  # Define windoes_size of D
img_lr, img_hr = prepare_clips('resources/test_clips/')  # Change this image name
tensor_lrs = PIL2Tensor(img_lr)
tensor_hrs = PIL2Tensor(img_hr)
cv2_lr = np.moveaxis(tensor_lrs.numpy(), 0, 2)
cv2_hr = np.moveaxis(tensor_hrs.numpy(), 0, 2)

b, c, orig_h, orig_w = tensor_lrs.shape

frame_index = math.ceil(b / 2) - 1

w, h, position_pil = click_select_position(img_hr[frame_index], window_size)

sigma = 1.2
fold = 50
l = 9
alpha = 0.5
attr_objective = attribution_objective(attr_grad, h, w, window=window_size)
gaus_blur_path_func = GaussianBlurPath(sigma, fold, l)
interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient(tensor_lrs.numpy(), model, attr_objective,
                                                                          gaus_blur_path_func, frame_index)

for index in range(b):
    grad_numpy, result = saliency_map(interpolated_grad_numpy[index], result_numpy[index])
    abs_normed_grad_numpy = grad_abs_norm(grad_numpy)
    saliency_image_abs = vis_saliency(abs_normed_grad_numpy, zoomin=4)
    saliency_image_kde = vis_saliency_kde(abs_normed_grad_numpy)
    blend_abs_and_input = cv2_to_pil(
        pil_to_cv2(saliency_image_abs) * (1.0 - alpha) + pil_to_cv2(img_lr[index].resize(img_hr[index].size)) * alpha)
    blend_kde_and_input = cv2_to_pil(
        pil_to_cv2(saliency_image_kde) * (1.0 - alpha) + pil_to_cv2(img_lr[index].resize(img_hr[index].size)) * alpha)
    pil = make_pil_grid(
        [position_pil,
         saliency_image_abs,
         blend_abs_and_input,
         blend_kde_and_input,
         Tensor2PIL(torch.clamp(torch.from_numpy(result), min=0., max=1.))]
    )
    # pil.show()
    pil.save("./results/vsr/" + str(index) + ".png")