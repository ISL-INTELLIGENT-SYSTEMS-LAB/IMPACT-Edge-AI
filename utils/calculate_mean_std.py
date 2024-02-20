import numpy as np
import os
from PIL import Image

def calculate_mean_std(IMG_DIR):

    mean = np.array([0., 0., 0.])
    std = np.array([0., 0., 0.])
    num_images = 0
    for root, dirs, files in os.walk(IMG_DIR):
        for file in files:
            img = Image.open(os.path.join(root, file))
            img = np.array(img).astype(np.float32) 
            mean += np.mean(img, axis=(0, 1))
            std += np.std(img, axis=(0, 1))
            num_images += 1
    mean /= num_images
    std /= num_images
    return mean, std

if __name__ == '__main__':
    img_dir = 'data/mars_images'
    MEAN, STD = calculate_mean_std(img_dir)
    print(f'Mean: {MEAN}, Std: {STD}')