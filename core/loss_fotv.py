# -*- coding: utf-8 -*-
# core/loss_fotv.py
#
# Fractional-order losses for CAMs:
#   - build_alpha_map_from_edges: adaptive α(x) with edge-band boosting
#   - frac_laplacian_loss:        || (-Δ)^{α/2} CAM ||_2^2
#   - frac_tv_loss:              TV_α(CAM) ≈ mean( w(x) * |∇^α CAM| )

import torch
import torch.nn.functional as F

__all__ = [
    "build_alpha_map_from_edges",
    "frac_laplacian_loss",
    "frac_tv_loss",
]

# ----------------- Frequency-domain utilities -----------------
def _fftfreq_2d(h, w, device):
    fy = torch.fft.fftfreq(h, d=1.0).to(device)
    fx = torch.fft.fftfreq(w, d=1.0).to(device)
    wy, wx = torch.meshgrid(fy, fx, indexing="ij")
    return wy, wx


def _abs_xi_pow_alpha(H, W, alpha, device, eps=1e-8):
    wy, wx = _fftfreq_2d(H, W, device)
    rho2 = wx**2 + wy**2 + eps
    return rho2 ** (alpha / 2.0)


def _riesz_fractional_grad(m, alpha=1.2, eps=1e-8):
    """
    Riesz fractional-order gradient (frequency-domain implementation).

    Args:
        m: Tensor of shape (B, C, H, W).

    Returns:
        gx, gy: Fractional gradients along x and y, each of shape (B, C, H, W).
    """
    B, C, H, W = m.shape
    wy, wx = _fftfreq_2d(H, W, m.device)
    rho = torch.sqrt(wx**2 + wy**2 + eps)
    kx = (1j * wx) * (rho ** (alpha - 1.0))
    ky = (1j * wy) * (rho ** (alpha - 1.0))
    M = torch.fft.fft2(m)
    Gx = torch.fft.ifft2(M * kx).real
    Gy = torch.fft.ifft2(M * ky).real
    return Gx, Gy


# ----------------- Adaptive α(x) construction -----------------
@torch.no_grad()
def build_alpha_map_from_edges(x, alpha_base=1.0, alpha_boost=0.3, band_dilate=3, eps=1e-6):
    """
    Build an adaptive fractional-order map α(x) from edge cues (with edge-band boosting).

    Args:
        x: Tensor of shape (B, C, H, W) or (B, 1, H, W) or (B, H, W).
           Can be an image or a CAM.
        alpha_base: Base fractional order.
        alpha_boost: Additional order added on edge bands.
        band_dilate: Max-pooling kernel size to dilate edges into an "edge band".
        eps: Small constant for numerical stability.

    Returns:
        alpha_map: Tensor of shape (B, 1, H, W), roughly in
                   [alpha_base, alpha_base + alpha_boost], clipped at >= 0.1.
    """
    if x.dim() == 3:
        x = x.unsqueeze(1)
    B, C, H, W = x.shape
    xg = x.mean(1, keepdim=True)

    # Sobel edge magnitude
    kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3) / 8.0
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(xg, kx, padding=1)
    gy = F.conv2d(xg, ky, padding=1)
    e = torch.sqrt(gx**2 + gy**2 + eps)

    # Normalize edges and optionally dilate to form an edge band
    emin = e.amin(dim=(-2, -1), keepdim=True)
    emax = e.amax(dim=(-2, -1), keepdim=True)
    e = (e - emin) / (emax - emin + eps)
    if band_dilate > 0:
        e = F.max_pool2d(e, kernel_size=band_dilate, stride=1, padding=band_dilate // 2)

    alpha_map = (alpha_base + alpha_boost * e).clamp_min(0.1)
    return alpha_map


# ----------------- Fractional Laplacian energy -----------------
def frac_laplacian_loss(cam: torch.Tensor, alpha_map: torch.Tensor, eps: float = 1e-8):
    """
    Fractional Laplacian energy: ||(-Δ)^{α/2} cam||_2^2.

    Args:
        cam: Tensor of shape (B, C, H, W) or (B, 1, H, W) or (B, H, W).
        alpha_map: Tensor of shape (B, 1, H, W).
        eps: Small constant for numerical stability.

    Notes:
        For efficiency and stability in large-batch training, the frequency kernel
        uses the spatial mean of α over the whole image (per sample).

    Returns:
        Scalar loss (Tensor).
    """
    if cam.dim() == 3:
        cam = cam.unsqueeze(1)
    B, C, H, W = cam.shape
    alpha_mean = alpha_map.mean(dim=(-2, -1), keepdim=True)  # (B, 1, 1, 1)

    loss = cam.new_tensor(0.0)
    for b in range(B):
        a = float(alpha_mean[b, 0, 0, 0].item())
        K = _abs_xi_pow_alpha(H, W, a, cam.device)
        M = torch.fft.fft2(cam[b])          # (C, H, W)
        frac = torch.fft.ifft2(M * K).real  # (C, H, W)
        loss = loss + (frac**2).mean()
    return loss / B


# ----------------- Fractional TV -----------------
def frac_tv_loss(cam: torch.Tensor, alpha_map: torch.Tensor, eps: float = 1e-8):
    """
    Fractional TV:
        TV_α(cam) ≈ mean( w(x) * |∇^α cam| )

    Where:
        ∇^α is approximated by the Riesz fractional gradient,
        and w(x) is a pixel-wise weight derived from alpha_map (heavier on edge bands).

    Args:
        cam: Tensor of shape (B, C, H, W) or (B, 1, H, W) or (B, H, W).
        alpha_map: Tensor of shape (B, 1, H, W).
        eps: Small constant for numerical stability.

    Returns:
        Scalar loss (Tensor).
    """
    if cam.dim() == 3:
        cam = cam.unsqueeze(1)

    # Use the global mean α as a scalar order for the frequency-domain operator
    a_scalar = float(alpha_map.mean().item())
    gx, gy = _riesz_fractional_grad(cam, alpha=a_scalar)
    mag = torch.sqrt(gx**2 + gy**2 + eps)  # (B, C, H, W)

    # Edge-band weighting: normalize alpha_map and clamp to a reasonable range
    w = (alpha_map / (alpha_map.mean() + 1e-8)).clamp(0.5, 2.0)  # (B, 1, H, W)
    if w.shape[1] != mag.shape[1]:
        w = w.repeat(1, mag.shape[1], 1, 1)

    return (w * mag).mean()
