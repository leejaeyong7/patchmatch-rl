import torch.nn.functional as NF
import matplotlib.pyplot as plt
import cv2
import numpy as np

def to_bin_image(img):
    '''
    Given 1x1xHxW depth image, returns nice visualization of the depth image
    '''
    bin_image = (img[0].detach().cpu().numpy() * 255).astype('uint8')
    bin_colored_image = cv2.applyColorMap(bin_image, cv2.COLORMAP_PARULA)

    return bin_colored_image[:, :, ::-1].transpose(2, 0, 1)
def to_depth_image(depth, ranges):
    '''
    Given 1x1xHxW depth image, returns nice visualization of the depth image
    '''
    d = depth[0].detach().cpu().numpy()
    min_d = ranges[0].cpu().numpy()
    max_d = ranges[1].cpu().numpy()
    depth_image = (((max_d - d) / (max_d - min_d)) * 255).astype('uint8')
    depth_colored_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_PARULA)
    depth_colored_image[(d == 0).repeat(3, 2)] = 0

    return depth_colored_image[:, :, ::-1].transpose(2, 0, 1)

def to_conf_image(conf):
    '''
    Given 1x1xHxW conf image, returns nice visualization of the depth image
    '''
    d = conf[0].detach().cpu().numpy()
    conf_image = (d * 255).astype('uint8')
    conf_colored_image = cv2.applyColorMap(conf_image, cv2.COLORMAP_PARULA)
    return conf_colored_image[:, :, ::-1].transpose(2, 0, 1)

def to_normal_image(normal):
    '''
    Given 1xHxWx3 normal image, returns nice visualization of the normal image
    '''
    normal_image = normal[0].detach().cpu().numpy().transpose(2, 0, 1)
    nn = (normal_image + 1) / 2
    return nn

def to_view_selection_images(vs, k=3):
    '''
    Given NxHxWx1 depth image, returns nice visualization of the vismap images
    '''
    view_map = NF.one_hot(vs.topk(k, dim=0).indices, len(vs)).sum(0).permute(3, 0, 1, 2).float()
    v = view_map.detach().cpu().numpy()
    vs_images = (v* 255).astype('uint8')
    translated =[]
    for vs_image in vs_images:
        vs_colored_image = cv2.applyColorMap(vs_image, cv2.COLORMAP_PARULA)

        translated.append(vs_colored_image[:, :, ::-1].transpose(2, 0, 1))
    return np.stack(translated)

def to_view_prob_images(vs):
    v = vs.detach().cpu().numpy()
    vs_images = (v* 255).astype('uint8')
    translated =[]
    for vs_image in vs_images:
        vs_colored_image = cv2.applyColorMap(vs_image, cv2.COLORMAP_PARULA)

        translated.append(vs_colored_image[:, :, ::-1].transpose(2, 0, 1))
    return np.stack(translated)
