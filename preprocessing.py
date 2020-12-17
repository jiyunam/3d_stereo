import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import time
from scipy.ndimage import gaussian_filter
from preprocessing_utils import *

import cv2


DESIRED_AVERAGE_IMAGE_INTENSITY = 150  # between 0 and 255
DESIRED_IMAGE_CONTRAST = 100  # between -127 and 127


def load_image(file_path):
    """
    Load the image and convert it to a 3D numpy array
    :param str file_path: The file path of the image
    :return: The array representation of the image
    :rtype: np.ndarray
    """
    #image = img.imread(file_path)
    image = cv2.imread(file_path)
    #print(np.shape(image))

    return image


def convert_to_greyscale(image_matrix):
    """
    Converts the image represented by the image matrix to greyscale if necessary
    :param np.ndarray image_matrix: The image to convert
    :return: The greyscale version of the input image
    :rtype: np.ndarray
    """
    if np.ndim(image_matrix) > 2:  # not a greyscale image
        # RGB weights to convert to intensity
        colour_to_intensity = np.asarray([0.2125, 0.7164, 0.0721])

        image_matrix = np.dot(image_matrix, colour_to_intensity)

    return image_matrix


def normalize_image_intensity(image_matrix):
    """

    :param image_matrix:
    :return:
    """
    average_intensity = np.mean(image_matrix)
    #print("Average intensity: {} | diff: {}".format(average_intensity, DESIRED_AVERAGE_IMAGE_INTENSITY - average_intensity))
    normalized_intensity_image_matrix = image_matrix + (DESIRED_AVERAGE_IMAGE_INTENSITY - average_intensity)

    return normalized_intensity_image_matrix


def modify_image_contrast(image_matrix):
    """

    :param image_matrix:
    :return:
    """
    brightness = 0  # TODO: test optimal brightness
    altered_contrast_image_matrix = image_matrix * (DESIRED_IMAGE_CONTRAST/127 + 1) - DESIRED_IMAGE_CONTRAST + brightness

    return altered_contrast_image_matrix


def image_normalization(image_matrix, sigma=1):
    """

    :param image_matrix:
    :return:
    """
    blurred_image_matrix = gaussian_filter(image_matrix, sigma=sigma)
    #display_image(blurred_image_matrix)

    normalized_intensity_image_matrix = normalize_image_intensity(blurred_image_matrix)
    #display_image(normalized_intensity_image_matrix)

    altered_contrast_image_matrix = modify_image_contrast(normalized_intensity_image_matrix)
    #display_image(altered_contrast_image_matrix)

    return altered_contrast_image_matrix


def sharpen_image(image_matrix):
    alpha = 5
    beta = 3
    delta = 1

    blurred_image_matrix = gaussian_filter(image_matrix, sigma=3)
    blurred_image_matrix_2 = gaussian_filter(image_matrix, sigma=5)
    blurred_image_matrix_3 = gaussian_filter(image_matrix, sigma=7)
    sharpened_image_matrix = image_matrix + \
                             alpha * (image_matrix - blurred_image_matrix) + \
                             beta * (image_matrix - blurred_image_matrix_2) + \
                             delta * (image_matrix - blurred_image_matrix_3)

    return sharpened_image_matrix


def random_subsample(image_matrix):
    minimum_area = 100000  # arbitrarily set this to an area equivalent to 100 pixels x 100 pixels
    minimum_leg_ratio = 2  # ensure that an extremely stretched out rectangle is not a valid area. Arbitrarily set this to 2

    minimum_x_delta = np.min([np.sqrt(minimum_area/minimum_leg_ratio), image_matrix.shape[0]])
    minimum_y_delta = np.min([np.sqrt(minimum_area/minimum_leg_ratio), image_matrix.shape[1]])

    maximum_x_delta = np.max([np.sqrt(minimum_area/minimum_leg_ratio) * minimum_leg_ratio, image_matrix.shape[0]])
    maximum_y_delta = np.max([np.sqrt(minimum_area/minimum_leg_ratio) * minimum_leg_ratio, image_matrix.shape[1]])

    x_delta = np.random.randint(minimum_x_delta, maximum_x_delta)
    y_delta = np.random.randint(minimum_y_delta, maximum_y_delta)

    x = np.random.randint(0, image_matrix.shape[0] - x_delta)
    y = np.random.randint(0, image_matrix.shape[1] - y_delta)

    return image_matrix[x:x + x_delta + 1, y:y + y_delta + 1]


def random_colour_alteration(image_matrix):
    max_fluctuation = 0.2

    random_colour_fluctuation = np.random.uniform((1-max_fluctuation), (1+max_fluctuation), size=(3))

    #print(image_matrix)
    altered_colour_image_matrix = (image_matrix * random_colour_fluctuation).astype(int)
    altered_colour_image_matrix[altered_colour_image_matrix >255] = 255
    #print(altered_colour_image_matrix)

    return altered_colour_image_matrix


def display_image(image_matrix, verbosity=1):
    if verbosity:
        plt.imshow(image_matrix, cmap='gray', clim=(0, 255))
        plt.show()


def sample_run():
    #TODO: implement this test run

    image_path = './data/dino/dino/dino{:04d}.png'
    saved_image_path = './processed_data/method{:02d}/dino{:04d}.png'
    image_number = 10
    verbosity = 0

    method = 10  # 0 = image preprocessing, 1 = image sampling, 2 = image colour warping , 10 = edge detection

    if method == 0:
        image_matrix = load_image(image_path.format(image_number))
        greyscale_image_matrix = convert_to_greyscale(image_matrix)
        display_image(greyscale_image_matrix, verbosity)

        normalized_image = image_normalization(greyscale_image_matrix)
        display_image(normalized_image, verbosity)

        resulting_image = sharpen_image(normalized_image)
        display_image(resulting_image)

    elif method == 1:
        image_matrix = load_image(image_path.format(image_number))
        display_image(image_matrix, verbosity)

        samples_to_take = 10
        subsamples = [random_subsample(image_matrix) for _ in range(samples_to_take)]

        for subsample in subsamples:
            display_image(subsample)

    elif method == 2:
        image_matrix = load_image(image_path.format(image_number))
        display_image(image_matrix, verbosity)

        warped_colour_image = random_colour_alteration(image_matrix)
        display_image(warped_colour_image)

    elif method == 10:
        # Edge detection
        image_matrix = load_image(image_path.format(image_number))
        greyscale_image_matrix = convert_to_greyscale(image_matrix)
        display_image(greyscale_image_matrix, verbosity)

        normalized_image = image_normalization(greyscale_image_matrix, sigma=3)
        display_image(normalized_image, verbosity)

        image_gradients = get_gradients(normalized_image)
        display_image(image_gradients, verbosity)
        print("MAX gradient {} | MIN gradient {}".format(np.max(image_gradients), np.min(image_gradients)))
        edge_matrix = intensity_gradient_threshold(image_gradients)
        display_image(edge_matrix)

        cv2.imwrite(saved_image_path.format(method, image_number), edge_matrix)

    elif method == 11:
        # Canny Edge detection
        image_matrix = load_image(image_path.format(image_number))
        greyscale_image_matrix = convert_to_greyscale(image_matrix)
        display_image(greyscale_image_matrix, verbosity)

        normalized_image = image_normalization(greyscale_image_matrix, sigma=3)
        display_image(normalized_image, verbosity)

        edge_matrix = cv2.Canny(normalized_image.astype(np.uint8), 50, 100, True)
        display_image(edge_matrix)

        cv2.imwrite(saved_image_path.format(method, image_number), edge_matrix)

    pass





if __name__ == '__main__':
    # do a test run on a sample
    sample_run()