import os
import nrrd
import numpy as np
from skimage.morphology import dilation, ball
import re  


def iterative_dilation(label_image, num_iterations, radius, ct_image, lower_bound, upper_bound):
    structuring_element = ball(radius)

    ct_mask = np.logical_and(ct_image >= lower_bound, ct_image <= upper_bound)

    dilated_image = label_image.copy()

    for i in range(num_iterations):
        dilated_image = dilation(dilated_image, structuring_element)
        dilated_image = np.logical_and(dilated_image, ct_mask)
        dilated_image = np.logical_or(dilated_image, label_image)

    return dilated_image


def extract_number(filename):

    match = re.search(r'(\d+)', filename)  
    if match:
        return int(match.group(1))  
    return -1 


ct_folder = r"D:\data_seg\image"  
label_folder = r"D:\data_seg\mask"
output_folder = r"D:\data_seg\math"

os.makedirs(output_folder, exist_ok=True)

dilation_radius = 2
lower_bound = -500
upper_bound = 200

ct_files = sorted(os.listdir(ct_folder), key=extract_number)
label_files = sorted(os.listdir(label_folder), key=extract_number)

for ct_file in ct_files:
    print("ct_file==",ct_file)
    if ct_file.endswith(".nrrd"):
        ct_name = os.path.splitext(ct_file)[0]
        ct_number = re.search(r'(\d+)', ct_name).group(1)
        ct_path = os.path.join(ct_folder, ct_file)

        ct_data, ct_header = nrrd.read(ct_path)

        for label_file in label_files:
            print("label_file==", label_file)
            label_number = re.search(r'(\d+)', label_file).group(1)

            if label_file.startswith(f"labelS{ct_number}") and label_number == ct_number:
                label_path = os.path.join(label_folder, label_file)

                label_data, label_header = nrrd.read(label_path)

                expanded_label = iterative_dilation(label_data, num_iterations=8, radius=1, ct_image=ct_data, lower_bound=lower_bound, upper_bound=upper_bound)

                expanded_label = expanded_label.astype(np.uint8)

                output_path = os.path.join(output_folder, label_file)
                nrrd.write(output_path, expanded_label, header=label_header)

                print(f"Processed {label_file} and saved to {output_path}")
