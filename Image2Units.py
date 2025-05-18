import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt


class Image2Units:

    def __init__(self, image_shape, shape_in_mm=(19, 25)):
        self.pixel_height_mm = shape_in_mm[0] / image_shape[0]
        self.pixel_width_mm = shape_in_mm[1] / image_shape[1]

    def get_mask_area_in_mm2(self, mask):
        return np.sum(mask > 0) * self.pixel_width_mm * self.pixel_height_mm

    def get_mask_length_in_mm(self, mask):
        skeleton = skeletonize(mask > 0)
        return np.sum(skeleton > 0) * np.sqrt(self.pixel_width_mm ** 2 + self.pixel_height_mm ** 2)


def create_hair_like_image(image_size=(256, 256), num_lines=2, line_thickness=7):
    """
    Generate a binary image with noodle-like shapes (curvy lines).

    Args:
    - image_size: Tuple representing the size of the image (height, width).
    - num_lines: Number of noodle-like curvy lines to generate.
    - line_thickness: Thickness of the noodle lines.

    Returns:
    - binary_image: A binary image with noodle-like shapes.
    """
    # Initialize a blank black image
    binary_image = np.zeros(image_size, dtype=np.uint8)

    # Randomly create lines resembling noodles
    for _ in range(num_lines):
        # Random starting and ending points
        start_x = np.random.randint(0, image_size[1])
        start_y = np.random.randint(0, image_size[0])
        end_x = np.random.randint(0, image_size[1])
        end_y = np.random.randint(0, image_size[0])

        # Generate a random number of bends and a random curvature
        num_bends = np.random.randint(2, 6)
        curvature = np.random.uniform(0.2, 1.0)

        # Create a curved line (approximating a noodle)
        x = np.linspace(start_x, end_x, num_bends * 10)
        y = np.linspace(start_y, end_y, num_bends * 10)

        # Add some random noise for curviness
        y += np.sin(np.linspace(0, np.pi * curvature, num_bends * 10)) * 10

        # Round the coordinates to create a valid line and ensure inside image bounds
        x = np.clip(np.round(x).astype(int), 0, image_size[1] - 1)
        y = np.clip(np.round(y).astype(int), 0, image_size[0] - 1)

        # Draw the line with a given thickness (using a simple thickening approach)
        for i in range(len(x)):
            for dx in range(-line_thickness, line_thickness + 1):
                for dy in range(-line_thickness, line_thickness + 1):
                    if 0 <= x[i] + dx < image_size[1] and 0 <= y[i] + dy < image_size[0]:
                        binary_image[y[i] + dy, x[i] + dx] = 1

    return binary_image


if __name__ == '__main__':
    binary_image = create_hair_like_image()

    # Apply skeletonization
    skeleton = skeletonize(binary_image)

    # Create an overlay image where skeleton is colored red and overlayed on original
    overlay_image = np.copy(binary_image)
    overlay_image = np.stack([overlay_image, overlay_image, overlay_image], axis=-1)  # RGB image
    overlay_image[np.where(skeleton)] = [1, 0, 0]  # Color the skeletonized pixels red

    # Ensure the overlay image is of type uint8 for compatibility with imshow
    overlay_image = (overlay_image * 255).astype(np.uint8)

    # Plot the three images: original, skeletonized, and the overlayed image
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(binary_image, cmap='gray')
    axes[0].set_title("Original Hair-like Binary Image")
    axes[0].axis('off')

    axes[1].imshow(skeleton, cmap='gray')
    axes[1].set_title("Skeletonized Image")
    axes[1].axis('off')

    axes[2].imshow(overlay_image)
    axes[2].set_title("Skeletonized Overlay on Original (Red)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()