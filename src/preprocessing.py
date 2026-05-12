import cv2
import numpy as np


def to_grayscale(image):

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_image(image, width, height):

    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def apply_blur(image, kernel_size=5):

    if kernel_size % 2 == 0 or kernel_size <= 0:
        raise ValueError("kernel_size debe ser un número impar positivo.")

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def adjust_brightness_contrast(image, alpha=1.0, beta=0):

    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def apply_threshold(image, thresh_value=127):

    if len(image.shape) != 2:
        raise ValueError("apply_threshold requiere una imagen en escala de grises (1 canal).")

    _, thresholded = cv2.threshold(
        image,
        thresh_value,
        255,
        cv2.THRESH_BINARY
    )

    return thresholded


def detect_edges(image, low=50, high=150):

    if len(image.shape) == 3:
        gray = to_grayscale(image)
    else:
        gray = image

    edges = cv2.Canny(gray, low, high)

    return edges


def full_pipeline(image, target_width=224, target_height=224):

    resized = resize_image(image, target_width, target_height)
    gray = to_grayscale(resized)
    blurred = apply_blur(gray, kernel_size=3)
    edges = detect_edges(blurred, low=50, high=150)

    return edges