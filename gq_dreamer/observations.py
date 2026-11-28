"""Model-side observation processing: image tensors, patch grids, MAE masks, motion weights, batching of env
observations."""
import numpy as np
import torch
import torch.nn.functional as F

OBS_KEYS = ("image",)                                                  # what the encoder reads


def autocast_dtype(device_type):
    """Current autocast dtype of this device, fp32 when autocast is off."""
    if torch.is_autocast_enabled(device_type):
        return torch.get_autocast_dtype(device_type)
    return torch.float32


# --- Frames ---------------------------------------------------------------------------------------------------------
def image_input(image, dtype=torch.float32):
    """uint8 images [N, 3, S, S] -> encoder input in [-1, 1] (no gradient needed: computed under no_grad)."""
    with torch.no_grad():
        return image.to(torch.float32, copy=True).mul_(2.0 / 255.0).sub_(1.0).to(dtype)


def image_target(image):
    """uint8 images [N, 3, S, S] -> fp32 reconstruction targets in [0, 1]."""
    return image.to(torch.float32, copy=True).div_(255.0)


# --- Patch grids ----------------------------------------------------------------------------------------------------
def patch_grid(img_size, patch):
    """(rows, columns) of the patch grid of an img_size x img_size frame."""
    assert img_size % patch == 0, f"patch size {patch} does not tile a {img_size}x{img_size} frame"
    return img_size // patch, img_size // patch


def n_patches(img_size, patch):
    """Number of patch tokens."""
    r, c = patch_grid(img_size, patch)
    return r * c


def patch_dim(patch, channels=3):
    """Values per patch."""
    return channels * patch * patch


def patchify(img, patch):
    """[N, C, H, W] -> [N, n_patches, C * p * p], row-major patches."""
    N, C, H, W = img.shape
    r, c = H // patch, W // patch
    x = img.reshape(N, C, r, patch, c, patch)
    return x.permute(0, 2, 4, 1, 3, 5).reshape(N, r * c, C * patch * patch)


def unpatchify(patches, patch, img_size, channels=3):
    """[N, n_patches, C * p * p] -> [N, C, S, S] (inverse of patchify)."""
    N = patches.shape[0]
    r = c = img_size // patch
    x = patches.reshape(N, r, c, channels, patch, patch)
    return x.permute(0, 3, 1, 4, 2, 5).reshape(N, channels, img_size, img_size)


def patch_pixel_mask(hide, patch, img_size):
    """Patch mask [N, n_patches] -> pixel mask [N, 1, S, S] (True on the pixels of the hidden patches)."""
    N = hide.shape[0]
    r = c = img_size // patch
    m = hide.view(N, 1, r, 1, c, 1).expand(N, 1, r, patch, c, patch)
    return m.reshape(N, 1, img_size, img_size)


def sample_hide(n, num_patches, mask_max=0.75, device="cpu"):
    """MAE mask [n, num_patches]: each frame hides a random share p ~ U(0, mask_max) of its patches,
    chosen uniformly."""
    p = torch.rand(n, 1, device=device) * mask_max
    return torch.rand(n, num_patches, device=device) < p


# --- Motion weights -------------------------------------------------------------------------------------------------
@torch.no_grad()
def motion_mask(cur, prev, threshold=0.05, dilate=1):
    """Pixels that changed since the previous frame: cur / prev fp32 [N, 3, S, S] in [0, 1] -> bool [N, 1, S, S]
    (max over channels > threshold, dilated by `dilate` pixels). A pixel-only object mask:
    balls, bullets and sprites move, the background mostly does not."""
    m = ((cur - prev).abs().amax(1, keepdim=True) > threshold).float()
    if dilate > 0:
        m = F.max_pool2d(m, 2 * dilate + 1, stride=1, padding=dilate)
    return m > 0


# --- Batching -------------------------------------------------------------------------------------------------------
def obs_to_tensors(obs_list, device):
    """List of env observations -> batched encoder inputs on device."""
    return {"image": torch.from_numpy(np.stack([o["image"] for o in obs_list])).to(device)}
