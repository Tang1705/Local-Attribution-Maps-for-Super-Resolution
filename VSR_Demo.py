import math

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

import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

global m, n
m, n = [], []


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        m.append(x)
        n.append(y)


# 'BASICVSR@Base' : BASICVSRNet
# 'ICONVSR@Base' : ICONVSRNet
# 'BASICVSRPP@Base' : BASICVSR_Plus_Plus_Net
# 'TTVSR@Base' : TTVSRNet
# 'PSRT@Base' : RethinkingAlignment
model = load_model('PSRT@Base')

window_size = 16  # Define windoes_size of D
img_lr, img_hr = prepare_clips('./REDS/')  # Change this image name
tensor_lrs = PIL2Tensor(img_lr)
tensor_hrs = PIL2Tensor(img_hr)
cv2_lr = np.moveaxis(tensor_lrs.numpy(), 0, 2)
cv2_hr = np.moveaxis(tensor_hrs.numpy(), 0, 2)

b, c, _, __ = tensor_lrs.shape

frame_index = math.ceil(b / 2) - 1

draw_img = pil_to_cv2(img_hr[frame_index])

cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", draw_img)

while len(m) == 0 and len(n) == 0:
    if cv2.getWindowProperty("image", cv2.WND_PROP_VISIBLE) <= 0:
        break
    cv2.waitKey(1)

w, h = m[0], n[0]
cv2.rectangle(draw_img, (w, h), (w + window_size, h + window_size), (0, 0, 255), 2)
cv2.imshow("image", draw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
position_pil = cv2_to_pil(draw_img)

sigma = 1.2
fold = 50
l = 9
alpha = 0.5
attr_objective = attribution_objective(attr_grad, h, w, window=window_size)
gaus_blur_path_func = GaussianBlurPath(sigma, fold, l)
interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient(tensor_lrs.numpy(), model, attr_objective,
                                                                          gaus_blur_path_func)

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
