import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageEnhance

def color_distort(image, settings=['contrast', 'sharpen', 'brighten', 'balance'], divisions=2):
	transforms = []

	if 'contrast' in settings:
		transforms.append(ImageEnhance.Contrast(Image))

	if 'sharpen' in settings:
		transforms.append(ImageEnhance.Sharpness(image))

	if 'brighten' in setting:
		transforms.append(ImageEnhance.Brightness(image))

	if 'balance' in settings:
		transforms.append(ImageEnhance.Color(image))

	transformed_images = []

	for transform in transform:
		for i in np.linspace(0.1, 1, divisions):
			transformed_images.append(transform.enhance(i))

	return transformed_images