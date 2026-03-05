import os
import cv2
import torch
import numpy as np
import torchvision.transforms as trn
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import concurrent.futures

import ctypes
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import skimage.color as skcolor
import skimage.util as skutil
from skimage.filters import gaussian
from scipy.ndimage import zoom as scizoom
from scipy.ndimage import map_coordinates
import warnings

warnings.simplefilter("ignore", UserWarning)

# ==========================================
# Data Loader Utils
# ==========================================

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm')

def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

# ==========================================
# Distortion Helpers
# ==========================================

def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

def plasma_fractal(mapsize=256, wibbledecay=3):
    assert (mapsize & (mapsize - 1) == 0), "mapsize must be a power of 2"
    maparray = np.empty((mapsize, mapsize), dtype=np.float32)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape).astype(np.float32)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        mapsize_current = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize_current:stepsize, stepsize // 2:mapsize_current:stepsize]
        ulgrid = maparray[0:mapsize_current:stepsize, 0:mapsize_current:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize_current:stepsize, stepsize // 2:mapsize_current:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize_current:stepsize, 0:mapsize_current:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()

def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    ch = int(np.ceil(h / zoom_factor))
    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    trim_top = (img.shape[0] - h) // 2
    return img[trim_top:trim_top + h, trim_top:trim_top + h]

# Wand library binding
wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double)

class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)

# ==========================================
# Frost Texture Cache
# ==========================================

_FROST_TEXTURES = []

def load_frost_textures():
    global _FROST_TEXTURES
    if _FROST_TEXTURES:
        return _FROST_TEXTURES

    base_dir = os.path.dirname(os.path.abspath(__file__))
    frost_dir = os.path.join(base_dir, 'frost_images')

    if not os.path.exists(frost_dir):
        raise FileNotFoundError(f"Frost image directory not found: {frost_dir}")

    valid_exts = ('.png', '.jpg', '.jpeg')
    for file in os.listdir(frost_dir):
        if file.lower().endswith(valid_exts):
            path = os.path.join(frost_dir, file)
            img = cv2.imread(path)
            if img is not None:
                _FROST_TEXTURES.append(img)

    if not _FROST_TEXTURES:
        raise RuntimeError(f"No valid image files found in directory: {frost_dir}")

    return _FROST_TEXTURES

# ==========================================
# Distortions
# ==========================================

def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.0
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255

def shot_noise(x, severity=1):
    c = [60, 25, 12, 5, 3][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.0
    return np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255

def impulse_noise(x, severity=1):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]
    x = skutil.random_noise(np.array(x, dtype=np.float32) / 255.0, mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255

def speckle_noise(x, severity=1):
    c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.0
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255

def gaussian_blur(x, severity=1):
    c = [1, 2, 3, 4, 6][severity - 1]
    x = gaussian(np.array(x, dtype=np.float32) / 255.0, sigma=c, channel_axis=-1)
    return np.clip(x, 0, 1) * 255

def glass_blur(x, severity=1):
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
    x = np.uint8(gaussian(np.array(x, dtype=np.float32) / 255.0, sigma=c[0], channel_axis=-1) * 255)
    
    h_img, w_img = x.shape[:2] # Remove hardcoded 224
    for i in range(c[2]):
        for h in range(h_img - c[1], c[1], -1):
            for w in range(w_img - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255.0, sigma=c[0], channel_axis=-1), 0, 1) * 255

def defocus_blur(x, severity=1):
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.0
    kernel = disk(radius=c[0], alias_blur=c[1])
    channels = [cv2.filter2D(x[:, :, d], -1, kernel) for d in range(3)]
    return np.clip(np.stack(channels, axis=-1), 0, 1) * 255

def motion_blur(x, severity=1):
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
    
    output = BytesIO()
    x.save(output, format='PNG')
    x = MotionImage(blob=output.getvalue())
    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

    x = cv2.imdecode(np.frombuffer(x.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)

    # Use len(shape) instead of checking against hardcoded tuples
    if len(x.shape) == 3: 
        return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
    else:  # greyscale to RGB
        return np.clip(np.stack([x, x, x], axis=-1), 0, 255)

def zoom_blur(x, severity=1):
    c = [np.arange(1, 1.11, 0.01),
         np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.02),
         np.arange(1, 1.26, 0.02),
         np.arange(1, 1.31, 0.03)][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.0
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255

def fog(x, severity=1):
    c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.0
    h_img, w_img = x.shape[:2]
    
    # Calculate nearest power of 2 for mapsize to prevent fractal generation failure
    mapsize = max(256, 2 ** int(np.ceil(np.log2(max(h_img, w_img)))))
    fractal = plasma_fractal(mapsize=mapsize, wibbledecay=c[1])[:h_img, :w_img][..., np.newaxis]
    
    max_val = x.max()
    x += c[0] * fractal
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255

def frost(x, severity=1):
    c = [(1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)][severity - 1]
    
    textures = load_frost_textures()
    idx = np.random.randint(len(textures))
    frost_img = textures[idx]
    
    h_img, w_img = np.array(x).shape[:2]
    
    # Security check: enlarge texture if it's smaller than the target image
    if frost_img.shape[0] <= h_img or frost_img.shape[1] <= w_img:
        frost_img = cv2.resize(frost_img, (max(w_img, frost_img.shape[1] + w_img), 
                                           max(h_img, frost_img.shape[0] + h_img)))
        
    x_start = np.random.randint(0, frost_img.shape[0] - h_img)
    y_start = np.random.randint(0, frost_img.shape[1] - w_img)
    frost_crop = frost_img[x_start:x_start + h_img, y_start:y_start + w_img][..., [2, 1, 0]]
    
    return np.clip(c[0] * np.array(x, dtype=np.float32) + c[1] * frost_crop, 0, 255)

def snow(x, severity=1):
    c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
         (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
         (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
         (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
         (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.
    h_img, w_img = x.shape[:2]
    
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])
    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = Image.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
    output = BytesIO()
    snow_layer.save(output, format='PNG')
    snow_layer = MotionImage(blob=output.getvalue())

    snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))
    snow_layer = cv2.imdecode(np.frombuffer(snow_layer.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED) / 255.
    snow_layer = snow_layer[..., np.newaxis]

    x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(h_img, w_img, 1) * 1.5 + 0.5)
    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255

def spatter(x, severity=1):
    c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
         (0.65, 0.3, 3, 0.68, 0.6, 0),
         (0.65, 0.3, 2, 0.68, 0.5, 0),
         (0.65, 0.3, 1, 0.65, 1.5, 1),
         (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.0
    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])
    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0

    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]

        color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1])), axis=2)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        x_bgra = cv2.cvtColor(x, cv2.COLOR_RGB2BGRA)
        return cv2.cvtColor(np.clip(x_bgra + m * color, 0, 1), cv2.COLOR_BGRA2RGB) * 255
    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0

        color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                42 / 255. * np.ones_like(x[..., :1]),
                                20 / 255. * np.ones_like(x[..., :1])), axis=2)
        color *= m[..., np.newaxis]
        x *= (1 - m[..., np.newaxis])
        return np.clip(x + color, 0, 1) * 255

def contrast(x, severity=1):
    c = [0.4, .3, .2, .1, .05][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.0
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255

def brightness(x, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.0
    x = skcolor.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    return np.clip(skcolor.hsv2rgb(x), 0, 1) * 255

def saturate(x, severity=1):
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.0
    x = skcolor.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    return np.clip(skcolor.hsv2rgb(x), 0, 1) * 255

def jpeg_compression(x, severity=1):
    c = [25, 18, 15, 10, 7][severity - 1]
    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    return Image.open(output)

def pixelate(x, severity=1):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    w, h = x.size
    x = x.resize((int(w * c), int(h * c)), Image.BOX)
    return x.resize((w, h), Image.BOX)

def elastic_transform(image, severity=1):
    c = [(224 * 2, 224 * 0.7, 224 * 0.1),
         (224 * 2, 224 * 0.08, 224 * 0.2),
         (224 * 0.05, 224 * 0.01, 224 * 0.02),
         (224 * 0.07, 224 * 0.01, 224 * 0.02),
         (224 * 0.12, 224 * 0.01, 224 * 0.02)][severity - 1]

    image = np.array(image, dtype=np.float32) / 255.0
    shape = image.shape
    shape_size = shape[:2]

    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]), c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]), c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255

CORRUPTIONS = {
    'gaussian_noise': gaussian_noise, 'shot_noise': shot_noise, 'impulse_noise': impulse_noise,
    'defocus_blur': defocus_blur, 'glass_blur': glass_blur, 'motion_blur': motion_blur, 'zoom_blur': zoom_blur, 
    'snow': snow, 'frost': frost, 'fog': fog, 'brightness': brightness, 
    'contrast': contrast, 'elastic_transform': elastic_transform, 'pixelate': pixelate, 'jpeg_compression': jpeg_compression
}

# ==========================================
# Multi-processing Orchestration
# ==========================================

def process_single_image(img_path, img_name, save_dir, transform, distort_method, severity):
    try:
        img = pil_loader(img_path)
        if transform is not None:
            img = transform(img)
            
        img_distorted = distort_method(img, severity)
        save_path = os.path.join(save_dir, img_name)
        
        if isinstance(img_distorted, Image.Image):
            img_distorted.save(save_path, quality=85, optimize=True)
        else:
            Image.fromarray(np.uint8(img_distorted)).save(save_path, quality=85, optimize=True)
            
    except Exception as e:
        print(f"Failed to process {img_name} at {img_path}: {e}")

def save_distorted_data(method_name, severity, data_path, save_path):
    if method_name not in CORRUPTIONS:
        raise ValueError(f"Unknown corruption method: {method_name}")
        
    distort_method = CORRUPTIONS[method_name]
    transform = trn.Compose([trn.Resize(256), trn.CenterCrop(224)])
    tasks = []
    
    # Use os.walk to robustly handle both nested folders and loose files
    for root, _, files in os.walk(data_path):
        rel_path = os.path.relpath(root, data_path)
        if rel_path == '.':
            rel_path = ''
            
        curr_save_dir = os.path.join(save_path, method_name, f'severity_{severity}', rel_path)
        os.makedirs(curr_save_dir, exist_ok=True)
        
        for name in sorted(files):
            if is_image_file(name):
                img_path = os.path.join(root, name)
                tasks.append((img_path, name, curr_save_dir, transform, distort_method, severity))

    if not tasks:
        print(f"No valid images found in {data_path}")
        return

    workers = min(os.cpu_count() or 4, 32)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_single_image, *task) for task in tasks]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc=f"{method_name} (Severity {severity})"):
            pass

# ==========================================
# CLI Entry Point
# ==========================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--corruption', type=str, default='all', choices=['all'] + list(CORRUPTIONS.keys()))
    parser.add_argument('--severity', type=int, default=5, choices=[1, 2, 3, 4, 5])
    parser.add_argument('--data_path', type=str, default='data/VGGSound/image_mulframe_test')
    parser.add_argument('--save_path', type=str, default='data/VGGSound/image_mulframe_test-C')
    args = parser.parse_args()

    corruptions_to_run = list(CORRUPTIONS.keys()) if args.corruption == 'all' else [args.corruption]

    for method in corruptions_to_run:
        save_distorted_data(method, args.severity, args.data_path, args.save_path)