import numpy as np


def precompute_BM_zzy(img, kHW, NHW, tauMatch):
    """
    :param img: input image (height, width)
    :param kHW: length of side of patch (patch size is kHW x kHW)
    :param NHW: number of most similar patches to retrieve for each reference patch
    :param tauMatch: threshold to determine whether two patches are similar
    """
    img = img.astype(np.float64)
    height, width = img.shape
    assert height % kHW == 0 and width % kHW == 0, "Image dimensions must be divisible by kHW"

    num_patches_y = height // kHW
    num_patches_x = width // kHW
    num_patches_total = num_patches_y * num_patches_x

    # Flatten the image into patches, non-overlapping, row first
    patch_list = []
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            patch = img[i * kHW:(i + 1) * kHW, j * kHW:(j + 1) * kHW]
            patch_list.append(patch)

    threshold = tauMatch * kHW * kHW
    diff_table = np.zeros((num_patches_total, num_patches_total), dtype=int)  # Store the squared differnces between any two patches

    for idx1, patch1 in enumerate(patch_list):
        for idx2, patch2 in enumerate(patch_list):
            # Compare A with B is the same as compare B with A, so just do once.
            # Also, do not compare a patch with itself, so here use '>'
            if idx2 > idx1:
                sum_diff_squared = np.sum((patch1 - patch2) * (patch1 - patch2))
                diff_table[idx1, idx2] = sum_diff_squared
                diff_table[idx2, idx1] = sum_diff_squared

    # Initialize return values
    relation_2d_indexed = np.ones((num_patches_y, num_patches_x, NHW, 2), dtype=int) * (-1)
    relation_1d_indexed = np.ones((num_patches_total, NHW), dtype=int) * (-1)
    similar_count_2d_indexed = np.zeros((num_patches_y, num_patches_x), dtype=int)
    similar_count_1d_indexed = np.zeros(num_patches_total, dtype=int)

    for ref_idx, ref_patch in enumerate(patch_list):
        y_of_ref_idx = ref_idx // num_patches_x
        x_of_ref_idx = ref_idx % num_patches_x

        valid_indices = np.where(diff_table[ref_idx] < threshold)[0]  # Get indices below the threshold
        valid_indices = valid_indices[valid_indices != ref_idx]  # Exclude the reference patch itself
        sorted_indices = valid_indices[np.argsort(diff_table[ref_idx][valid_indices])]  # Sort valid indices

        # Some reference patch may not have enough similar patches
        if len(sorted_indices >= NHW):
            most_similar_indices = sorted_indices[:NHW]
        else:
            most_similar_indices = sorted_indices

        similar_count_2d_indexed[y_of_ref_idx, x_of_ref_idx] = len(most_similar_indices)
        similar_count_1d_indexed[ref_idx] = len(most_similar_indices)

        for i, sim_idx in enumerate(most_similar_indices):
            y_of_sim_idx = sim_idx // num_patches_x
            x_of_sim_idx = sim_idx % num_patches_x

            relation_2d_indexed[y_of_ref_idx, x_of_ref_idx, i, 0] = y_of_sim_idx
            relation_2d_indexed[y_of_ref_idx, x_of_ref_idx, i, 1] = x_of_sim_idx
            relation_1d_indexed[ref_idx, i] = sim_idx

    return relation_2d_indexed, relation_1d_indexed, similar_count_2d_indexed, similar_count_1d_indexed


if __name__ == '__main__':
    import cv2
    import numpy as np
    import random

    def visualize_patches(img, kHW, NHW, tauMatch, border_thickness, output_path):
        """
        Visualize the patches and their similarities, then save the generated image.
        :param img: Input image (color (h, w, 3), or grayscale (h, w)).
        :param kHW: Length of side of patch (patch size is kHW x kHW).
        :param NHW: Number of most similar patches to display.
        :param tauMatch: Threshold to determine whether two patches are similar.
        :param border_thickness: Thickness of the border between patches.
        :param output_path: Path to save the visualized image.
        """

        relation_2d_indexed, _, _, _ = precompute_BM_zzy(img, kHW, NHW, tauMatch)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        height, width, _ = img.shape
        num_patches_y = height // kHW
        num_patches_x = width // kHW

        # Calculate the size of the image with added boundaries
        new_height = height + (num_patches_y + 1) * border_thickness
        new_width = width + (num_patches_x + 1) * border_thickness

        # Create a new image with a larger size to accommodate the boundaries
        img_visual = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255  # White background

        # Copy the original image into the new image
        for i in range(0, num_patches_y):
            for j in range(0, num_patches_x):
                img_visual[border_thickness + i * (kHW + border_thickness):border_thickness + i * (kHW + border_thickness) + kHW,
                           border_thickness + j * (kHW + border_thickness):border_thickness + j * (kHW + border_thickness) + kHW, :] \
                    = img[i * kHW:(i + 1) * kHW, j * kHW:(j + 1) * kHW, :]


        # Randomly select 4 reference patches
        yx_of_ref_idx = random.sample([(y, x) for y in range(num_patches_y) for x in range(num_patches_x)], 4)
        # Bounding box color for each of the reference blocks, and their corresponding similar blocks
        ref_colors = [(0, 0, 150), (0, 110, 0), (150, 0, 0), (0, 0, 0)]

        for i, (ref_y, ref_x) in enumerate(yx_of_ref_idx):
            # Get the coordinates of the reference patch and its similar patches
            yx_of_sim_idx = relation_2d_indexed[ref_y, ref_x]

            # Mark the reference patch's boundary, ENDPOINTS ARE (X, Y) DENOTED, NOT (Y, X) IN CV2.RECTANGLE!!!
            ref_top_left = (ref_x * (kHW + border_thickness), ref_y * (kHW + border_thickness))
            ref_bottom_right = (ref_x * (kHW + border_thickness) + kHW + border_thickness, ref_y * (kHW + border_thickness) + kHW + border_thickness)
            cv2.rectangle(img_visual, ref_top_left, ref_bottom_right, ref_colors[i], border_thickness)
            cv2.drawMarker(img_visual, ((ref_top_left[0] + ref_bottom_right[0]) // 2, (ref_top_left[1] + ref_bottom_right[1]) // 2), ref_colors[i], cv2.MARKER_TILTED_CROSS, 4, border_thickness)

            # Mark the similar patches' boundaries, ENDPOINTS ARE (X, Y) DENOTED, NOT (Y, X) IN CV2.RECTANGLE!!!
            for (sim_y, sim_x) in yx_of_sim_idx:
                if sim_y != -1 and sim_x != -1:
                    sim_top_left = (sim_x * (kHW + border_thickness), sim_y * (kHW + border_thickness))
                    sim_bottom_right = (sim_x * (kHW + border_thickness) + kHW + border_thickness, sim_y * (kHW + border_thickness) + kHW + border_thickness)
                    cv2.rectangle(img_visual, sim_top_left, sim_bottom_right, ref_colors[i], border_thickness)

        cv2.imwrite(output_path, img_visual)
        print(f"Image saved to {output_path}")

    img = cv2.imread('test_data/image/house.png', cv2.IMREAD_GRAYSCALE)
    kHW = 8  # Patch size of 8x8
    NHW = 15  # Number of similar patches to visualize
    tauMatch = 400  # Threshold for similarity

    visualize_patches(img, kHW, NHW, tauMatch, 1, 'tmp/bm_visualized.png')
