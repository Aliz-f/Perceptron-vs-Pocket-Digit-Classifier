from scipy.ndimage import sobel
import numpy as np

def extract_edge_density(image_row):
    # Get the image (16x16)
    image = image_row[1:].values.reshape(16, 16)

    # Sobel filter to detect edges
    sobel_x = sobel(image, axis=0)  # Horizontal edges
    sobel_y = sobel(image, axis=1)  # Vertical edges
    edge_magnitude = np.hypot(sobel_x, sobel_y)  # Magnitude of the edges

    # Edge density: ratio of edge pixels to total pixels (threshold to detect edges)
    edge_density = np.sum(edge_magnitude > 0.1) / 256

    return edge_density


def extract_aspect_ratio(image_row):
    # Get the image (16x16)
    image = image_row[1:].values.reshape(16, 16)

    # Find the bounding box of the digit (non-zero pixels)
    rows = np.any(image != 0, axis=1)
    cols = np.any(image != 0, axis=0)

    # Height and width of the bounding box
    height = np.sum(rows)
    width = np.sum(cols)

    # Aspect ratio (height/width)
    if width == 0:  # Avoid division by zero
        return 0
    aspect_ratio = height / width

    return aspect_ratio


def extract_bounding_box_compactness(image_row):
    # Get the image (16x16)
    image = image_row[1:].values.reshape(16, 16)

    # Find the bounding box of the digit (non-zero pixels)
    rows = np.any(image != 0, axis=1)
    cols = np.any(image != 0, axis=0)

    # Height and width of the bounding box
    height = np.sum(rows)
    width = np.sum(cols)

    # Area of bounding box
    bounding_box_area = height * width

    # Number of foreground pixels (non-zero pixels)
    num_foreground_pixels = np.sum(image != 0)

    # Bounding box compactness
    if bounding_box_area == 0:  # Avoid division by zero
        return 0
    compactness = num_foreground_pixels / bounding_box_area

    return compactness


def extract_center_of_mass(image_row):
    # Get the image (16x16)
    image = image_row[1:].values.reshape(16, 16)

    # Pixel coordinates
    y_coords, x_coords = np.indices(image.shape)

    # Total intensity (mass)
    total_mass = np.sum(image)

    # Compute centroid (weighted average of x and y coordinates)
    if total_mass == 0:  # Avoid division by zero
        return (0, 0)

    x_center_of_mass = np.sum(x_coords * image) / total_mass
    y_center_of_mass = np.sum(y_coords * image) / total_mass

    return (x_center_of_mass, y_center_of_mass)
