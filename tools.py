#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
a tool for data_sampler.py
"""

import numpy as np
import torch

def tensor_plane_former(a, b, c, h, w):
    # a, b, c = a.data.numpy(), b.data.numpy(), c.data.numpy()

    N = a.size(0)
    x, y = np.linspace(0, 1, w), np.linspace(0, 1, h)
    xx, yy = np.meshgrid(x, y)
    # xx, yy = torch.from_numpy(xx.astype(np.float)), torch.from_numpy(yy.astype(np.float))
    xx, yy = torch.from_numpy(xx), torch.from_numpy(yy)

    plane_list = []
    for i in range(N):
        zz = xx*(a[i, 0].type(torch.DoubleTensor)) + yy*(b[i, 0].type(torch.DoubleTensor)) \
             + c[i, 0].type(torch.DoubleTensor)
        plane_list.append(torch.unsqueeze(zz, 0))
    plane = torch.cat(plane_list, 0)
    return plane


def tensor_gradient(tensor):
    h, w = tensor.size(2), tensor.size(3)
    h_g = torch.cat((tensor[:, :, :-1, :] - tensor[:, :, 1:, :], tensor[:, :, h-1:h, :] - tensor[:, :, 0:1, :]), 2)
    w_g = torch.cat((tensor[:, :, :, :-1] - tensor[:, :, :, 1:], tensor[:, :, :, w-1:w] - tensor[:, :, :, 0:1]), 3)
    g = torch.sqrt(torch.pow(h_g, 2) + torch.pow(w_g, 2))
    return g

def xi(x):
    def theta_t(xxx):
        return np.arcsin(np.sin(xxx) / 1.474)
    theta_plus = x + theta_t(x)
    theta_minus = x - theta_t(x)
    return 2*(np.sin(theta_minus) ** 2)/(np.sin(theta_minus)**2 + np.sin(theta_plus)**2) + \
        2*(np.tan(theta_minus) ** 2)/(np.tan(theta_minus)**2 + np.tan(theta_plus)**2)

def tensor_xi(x):
    def theta_t(xxx):
        return torch.asin(torch.sin(xxx) / 1.474)
    theta_plus = x + theta_t(x)
    theta_minus = x - theta_t(x)
    return 2*(torch.sin(theta_minus) ** 2)/(torch.sin(theta_minus)**2 + torch.sin(theta_plus)**2) + \
        2*(torch.tan(theta_minus) ** 2)/(torch.tan(theta_minus)**2 + torch.tan(theta_plus)**2)

def zeta(x, phi_perp, phi_0):
    def theta_t(xxx):
        return np.arcsin(np.sin(xxx) / 1.474)
    theta_plus = x + theta_t(x)
    theta_minus = x - theta_t(x)
    return 2 * (np.sin(theta_minus) ** 2) / (np.sin(theta_minus) ** 2 + np.sin(theta_plus) ** 2) * (np.cos(phi_0 - phi_perp))**2 + \
        2 * (np.tan(theta_minus) ** 2) / (np.tan(theta_minus) ** 2 + np.tan(theta_plus) ** 2) * (np.sin(phi_0 - phi_perp))**2

def tensor_zeta(x, phi_perp, phi_0):
    def theta_t(xxx):
        return torch.asin(torch.sin(xxx) / 1.474)
    theta_plus = x + theta_t(x)
    theta_minus = x - theta_t(x)
    return 2 * (torch.sin(theta_minus) ** 2) / (torch.sin(theta_minus) ** 2 + torch.sin(theta_plus) ** 2) * (torch.cos(phi_0 - phi_perp))**2 + \
        2 * (torch.tan(theta_minus) ** 2) / (torch.tan(theta_minus) ** 2 + torch.tan(theta_plus) ** 2) * (torch.sin(phi_0 - phi_perp))**2

# def alpha_func(x):
#     x = np.deg2rad(x)
#     def theta_t(xxx):
#         return np.arcsin(np.sin(xxx) / 1.474)
#     theta_plus = x + theta_t(x)
#     theta_minus = x - theta_t(x)
#     return (np.sin(theta_minus) ** 2)/(np.sin(theta_minus)**2 + np.sin(theta_plus)**2) + \
#            (np.tan(theta_minus) ** 2)/(np.tan(theta_minus)**2 + np.tan(theta_plus)**2)


def MSE(img1, img2):
    img1, img2 = img1*255, img2*255
    try:
        h, w, c = img1.shape[0], img2.shape[1], img1.shape[2]
        N = h * w * c
        mse = ((img1 - img2)**2).sum() / N
    except IndexError:
        h, w = img1.shape[0], img2.shape[1]
        N = h * w
        mse = ((img1 - img2)**2).sum() / N

    return mse


def PSNR(img1, img2):
    mse = MSE(img1, img2)
    psnr = 10 * np.log10(255.**2/mse)
    return psnr


def SSIM(img1, img2):
    L = 255
    img1, img2 = img1*L, img2*L
    k1, k2 = 0.01, 0.03
    c1, c2 = (k1*L)**2, (k2*L)**2
    try:
        h, w, c = img1.shape[0], img2.shape[1], img1.shape[2]
        ssim_list = [0, 0, 0]
        for i in range(c):
            img1_mean = img1[:, :, i].sum()/(h*w)
            img2_mean = img2[:, :, i].sum()/(h*w)
            img1_sigma = np.sqrt(((img1[:, :, i] - img1_mean) ** 2).sum() / (h * w - 1))
            img2_sigma = np.sqrt(((img2[:, :, i] - img2_mean) ** 2).sum() / (h * w - 1))
            sigma_12 = ((img1[:, :, i] - img1_mean) * (img2[:, :, i] - img2_mean)).sum()/(h * w - 1)
            ssim_list[i] = (2*img1_mean*img2_mean+c1)*(2*sigma_12+c2)/(img1_mean**2+img2_mean**2+c1)/(img1_sigma**2+img2_sigma**2+c2)
            
        ssim = (ssim_list[0] + ssim_list[1] + ssim_list[2])/3.
    except IndexError:
        h, w = img1.shape[0], img2.shape[1]
        img1_mean = img1.sum() / (h * w)
        img2_mean = img2.sum() / (h * w)
        img1_sigma = np.sqrt(((img1 - img1_mean) ** 2).sum() / (h * w - 1))
        img2_sigma = np.sqrt(((img2 - img2_mean) ** 2).sum() / (h * w - 1))
        sigma_12 = ((img1 - img1_mean) * (img2 - img2_mean)).sum() / (h * w - 1)
        ssim = (2 * img1_mean * img2_mean + c1) * (2 * sigma_12 + c2) / (
                    img1_mean ** 2 + img2_mean ** 2 + c1) / (img1_sigma ** 2 + img2_sigma ** 2 + c2)
    return ssim


def cal_aoi_3d(x, y, z, nx, ny, nz):
    aoi_rad = np.arccos(np.abs(x*nx+y*ny+z*nz)/(np.sqrt(x**2+y**2+z**2)*np.sqrt(nx**2+ny**2+nz**2)))
    return aoi_rad


# def cal_aoi_2d(x, y, nx, ny):
#     aoi_rad = np.arccos(np.abs(x*nx+y*ny)/(np.sqrt(x**2+y**2)*np.sqrt(nx**2+ny**2)))
#     aoi_rad[np.isnan(aoi_rad)] = 0
#     return np.rad2deg(aoi_rad)


def cal_tensor_aoi_3d(x, y, z, nx, ny, nz):
    aoi_rad = torch.acos(torch.abs(x * nx + y * ny + z * nz) / (torch.sqrt(x ** 2 + y ** 2 + z ** 2) * torch.sqrt(nx ** 2 + ny ** 2 + nz ** 2)))
    return aoi_rad


def cal_phi(x, y, z, nx, ny, nz):
    poi_normal_x, poi_normal_y = y * nz - ny * z, z * nx - x * nz
    phi_rad = np.arctan2(poi_normal_y, poi_normal_x)
    return phi_rad

def cal_tensor_phi(x, y, z, nx, ny, nz):
    poi_normal_x, poi_normal_y = y * nz - ny * z, z * nx - x * nz
    phi_rad = torch.atan2(poi_normal_y, poi_normal_x)
    return phi_rad

def para_mesh_matrix_former(h, w, alpha, beta):
    z0 = 1
    c_x, c_y = w / 2, h / 2
    f_x, f_y = w * 1.4, w * 1.4
    kk = np.tan(alpha)
    xx, yy = np.meshgrid(np.linspace(0, w - 1, w), np.linspace(0, h - 1, h))
    denominator = (f_x * np.cos(beta) + kk * (xx - c_x) - np.sin(beta) * (yy - c_y)) / z0 / np.cos(beta)
    glass_x = (xx - c_x) / denominator
    glass_y = (yy - c_y) / denominator
    glass_z = f_x / denominator
    normal_x, normal_y, normal_z = kk, -np.sin(beta), np.cos(beta)
    AOI = cal_aoi_3d(glass_x, glass_y, glass_z, normal_x, normal_y, normal_z)
    AOI[AOI == 0] = 0.000001
    phi = cal_phi(glass_x, glass_y, glass_z, normal_x, normal_y, normal_z)
    return AOI, phi

def tensor_para_mesh_matrix_former(h, w, alpha, beta):
    z0 = 1
    c_x, c_y = w / 2, h / 2
    f_x, f_y = w * 1.4, w * 1.4
    kk = torch.tan(alpha)
    xx, yy = torch.meshgrid([torch.linspace(0, w - 1, steps=w), 
                             torch.linspace(0, h - 1, steps=h)])
    xx, yy = xx.transpose(0, 1), yy.transpose(0, 1)
    denominator = (f_x * torch.cos(beta) + kk * (xx - c_x) - torch.sin(beta) * (yy - c_y)) / torch.cos(beta)
    glass_x = (xx - c_x) / denominator
    glass_y = (yy - c_y) / denominator
    glass_z = f_x / denominator
    normal_x, normal_y, normal_z = kk, -torch.sin(beta), torch.cos(beta)
    AOI = cal_tensor_aoi_3d(glass_x, glass_y, glass_z, normal_x, normal_y, normal_z)
    AOI[AOI == 0] = 0.000001
    phi = cal_tensor_phi(glass_x, glass_y, glass_z, normal_x, normal_y, normal_z)
    return AOI, phi

