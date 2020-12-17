import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.signal import convolve2d

GRADIENT_CAP_MULTIPLIER = 8


def get_gradients(image_matrix):
    """
    Finds the gradients of the intensity of an image.
    :param np.ndarray image_matrix: The image to scan over
    :return: The array of intensity gradients of the image
    :rtype: np.ndarray
    """

    sobel_operator_gx = np.asarray([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_operator_gy = np.asarray([[-1,-2,-1],[0,0,0],[1,2,1]])
    # sobel operator requires a padding of 2 pixels on every side of the image

    gradient_x = convolve2d(image_matrix, sobel_operator_gx)
    gradient_y = convolve2d(image_matrix, sobel_operator_gy)

    total_gradient = np.sqrt(gradient_x**2 + gradient_y**2)

    mean_gradient = np.mean(total_gradient)

    total_gradient[total_gradient>mean_gradient*GRADIENT_CAP_MULTIPLIER] = mean_gradient*GRADIENT_CAP_MULTIPLIER

    return total_gradient


def visualize_image(image_matrix, mode='gray'):
    """
    Display the image via matplotlib.pyplot
    :param image_matrix: The image matrix to display
    :param mode: Colour mode to use, defaults to 'gray'
    :return:
    """
    plt.imshow(image_matrix, cmap=mode)
    plt.colorbar()
    plt.show()


def intensity_gradient_threshold(intensity_gradient_matrix):
    """
    Determines the edges from the Intensity Gradient Matrix
    :param np.ndarray intensity_gradient_matrix: The matrix containing the intensity gradients of the image
    :return: The edge-mapped image matrix
    :rtype: np.ndarray
    """

    ACCEPTABLE_ERROR_THRESHOLD = 0.0001

    igm = copy.deepcopy(intensity_gradient_matrix)
    # Initial threshold is the mean intensity of the image.
    threshold = np.mean(igm)
    threshold_old = 0

    iteration_index = 0

    while np.abs(threshold - threshold_old) > ACCEPTABLE_ERROR_THRESHOLD:
        print("Iteration {} of gradient thresholding. Threshold difference is {}".format(iteration_index, threshold-threshold_old))
        print("Mean threshold {}".format(threshold))
        lower_gradients_mean = np.mean(igm[igm<threshold])
        upper_gradients_mean = np.mean(igm[igm>=threshold])

        threshold_old = threshold
        threshold = (lower_gradients_mean + upper_gradients_mean) / 2.0
        iteration_index += 1

    igm[igm<threshold] = 0
    igm[igm>=threshold] = 255

    return igm

