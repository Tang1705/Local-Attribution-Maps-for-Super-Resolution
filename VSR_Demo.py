import torch, cv2, os, sys, numpy as np, matplotlib.pyplot as plt
from PIL import Image
from ModelZoo.utils import load_as_tensor, Tensor2PIL, PIL2Tensor, _add_batch_one
from ModelZoo import get_model, load_model, print_network
from SaliencyModel.utils import vis_saliency, vis_saliency_kde, click_select_position, grad_abs_norm, grad_norm, \
    prepare_clips, make_pil_grid, blend_input
from SaliencyModel.utils import cv2_to_pil, pil_to_cv2, gini
from SaliencyModel.attributes import attr_grad
from SaliencyModel.VSR_BackProp import I_gradient, attribution_objective, Path_gradient
from SaliencyModel.VSR_BackProp import saliency_map_PG as saliency_map
from SaliencyModel.VSR_BackProp import GaussianBlurPath
from SaliencyModel.utils import grad_norm, IG_baseline, interpolation, isotropic_gaussian_kernel

# 'BASICVSR@Base' : BASICVSRNet
# 'ICONVSR@Base' : ICONVSRNet
# 'BASICVSRPP@Base' : BASICVSR_Plus_Plus_Net
model = load_model('BASICVSR@Base')

window_size = 64  # Define windoes_size of D
img_lr, img_hr = prepare_clips('./test_clips/')  # Change this image name
tensor_lrs = PIL2Tensor(img_lr)
tensor_hrs = PIL2Tensor(img_hr)
cv2_lr = np.moveaxis(tensor_lrs.numpy(), 0, 2)
cv2_hr = np.moveaxis(tensor_hrs.numpy(), 0, 2)

w = 920  # The x coordinate of your select patch, 125 as an example
h = 220  # The y coordinate of your select patch, 160 as an example
# And check the red box
# Is your selected patch this one? If not, adjust the `w` and `h`.

sigma = 1.2
fold = 2
l = 9
alpha = 0.5
attr_objective = attribution_objective(attr_grad, h, w, window=window_size)
gaus_blur_path_func = GaussianBlurPath(sigma, fold, l)
interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient(tensor_lrs.numpy(), model, attr_objective,
                                                                          gaus_blur_path_func)

draw_img = pil_to_cv2(img_hr[2])
cv2.rectangle(draw_img, (w, h), (w + window_size, h + window_size), (0, 0, 255), 2)
position_pil = cv2_to_pil(draw_img)

b, c, _, __ = tensor_lrs.shape
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
    pil.show()
    # pil.save("./results/vsr/" + str(index) + ".png")
