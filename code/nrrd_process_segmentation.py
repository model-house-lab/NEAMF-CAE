from __future__ import print_function, division
import numpy as np
import cv2
import os



def subimage_generator(image, mask, math_image, patch_block_size, numberxy, numberz, depth, subnumber):
    width = np.shape(image)[1]
    height = np.shape(image)[2]
    imagez = np.shape(image)[0]
    block_width = np.array(patch_block_size)[1]
    block_height = np.array(patch_block_size)[2]
    blockz = np.array(patch_block_size)[0]
    stridewidth = (width - block_width) // numberxy
    strideheight = (height - block_height) // numberxy
    stridez = (imagez - blockz) // numberz

    if (stridez == 0):
        stridez = 1

    if stridez >= 1 and stridewidth >= 1 and strideheight >= 1:
        step_width = width - (stridewidth * numberxy + block_width)
        step_width = step_width // 2
        step_height = height - (strideheight * numberxy + block_height)
        step_height = step_height // 2
        step_z = imagez - (stridez * numberz + blockz)
        step_z = step_z // 2

        hr_samples_list = []
        hr_mask_samples_list = []
        hr_math_samples_list = []

        for z in range(step_z, numberz * (stridez + 1) + step_z, numberz):
            for x in range(step_width, numberxy * (stridewidth + 1) + step_width, numberxy):
                for y in range(step_height, numberxy * (strideheight + 1) + step_height, numberxy):

                    temp1 = (mask[z:z + blockz, x:x + block_width, y:y + block_height] == 255 - subnumber).sum()
                    if temp1 > 0:
                        hr_samples_list.append(
                            (image[z:z + blockz, x:x + block_width, y:y + block_height], x, y, z + depth))
                        hr_mask_samples_list.append(
                            (mask[z:z + blockz, x:x + block_width, y:y + block_height], x, y, z + depth))
                        hr_math_samples_list.append(
                            (math_image[z:z + blockz, x:x + block_width, y:y + block_height], x, y,
                             z + depth))
        print("len(hr_samples_list) =", len(hr_samples_list))


        hr_samples = np.array(hr_samples_list).reshape((len(hr_samples_list), 4))
        hr_mask_samples = np.array(hr_mask_samples_list).reshape((len(hr_mask_samples_list), 4))
        hr_math_samples = np.array(hr_math_samples_list).reshape((len(hr_math_samples_list), 4))

        return hr_samples, hr_mask_samples, hr_math_samples

    else:
        print("stridez =", stridez, "stridewidth =", stridewidth, "strideheight =", strideheight)
        nb_sub_images = 1 * 1 * 1
        hr_samples = np.zeros(shape=(nb_sub_images, blockz, block_width, block_height), dtype=np.float)
        hr_mask_samples = np.zeros(shape=(nb_sub_images, blockz, block_width, block_height), dtype=np.float)
        hr_math_samples = np.zeros(shape=(nb_sub_images, blockz, block_width, block_height), dtype=np.float)

        rangz = min(imagez, blockz)
        rangwidth = min(width, block_width)
        rangheight = min(height, block_height)

        hr_samples[0, 0:rangz, 0:rangwidth, 0:rangheight] = image[0:rangz, 0:rangwidth, 0:rangheight]
        hr_mask_samples[0, 0:rangz, 0:rangwidth, 0:rangheight] = mask[0:rangz, 0:rangwidth, 0:rangheight]
        hr_math_samples[0, 0:rangz, 0:rangwidth, 0:rangheight] = math_image[0:rangz, 0:rangwidth, 0:rangheight]

        return hr_samples, hr_mask_samples, hr_math_samples  # Return math samples as well


def make_patch(image, mask, math_image, patch_block_size, numberxy, numberz, depth, subnumber):

    image_subsample, mask_subsample, math_subsample = subimage_generator(image=image, mask=mask, math_image=math_image,
                                                                         patch_block_size=patch_block_size, numberxy=numberxy,
                                                                         numberz=numberz, depth=depth, subnumber=subnumber)
    return image_subsample, mask_subsample, math_subsample

import re


def extract_depth(file_name):
    match = re.search(r'_z(\d+)', file_name)
    if match:
        depth = int(match.group(1))
        return depth
    else:
        return None


def sort_key(file_name):
    return int(''.join(filter(str.isdigit, file_name)))


def gen_image_mask(srcimg, seg_image, math_image, index, shape, numberxy, numberz, trainImage, trainMask, trainMath,
                   depth):
    subnumber = 0
    break_flag_1 = False
    for subnumber in range(1, 255):
        sub_srcimages, sub_liverimages, sub_mathimages = make_patch(srcimg, seg_image, math_image,
                                                                    patch_block_size=shape, numberxy=numberxy,
                                                                    numberz=numberz, depth=depth, subnumber=subnumber)
        samples, imagez = np.shape(sub_srcimages)[0], 32
        if samples == 0:
            break

        for j in range(samples):
            sub_images, x, y, z_coordinate = sub_srcimages[j]
            sub_masks, _, _, _ = sub_liverimages[j]
            sub_math, _, _, _ = sub_mathimages[j]
            sub_masks = sub_masks.astype(np.float32)
            sub_masks = np.clip(sub_masks, 0, 255).astype('uint8')


            if np.max(sub_masks[:, :, :]) != 0:
                count = np.sum(sub_masks[:, :, :] == 255 - subnumber)
                if (count <= 30):
                    print("count=", count, "so continue")
                    continue
                filepath = f"{trainImage}\\{index}_{subnumber}_{j}_x{y}_y{x}_z{z_coordinate}_{count}\\"
                filepath2 = f"{trainMask}\\{index}_{subnumber}_{j}_x{y}_y{x}_z{z_coordinate}_{count}\\"
                filepath3 = f"{trainMath}\\{index}_{subnumber}_{j}_x{y}_y{x}_z{z_coordinate}_{count}\\"


                if not os.path.exists(filepath) and not os.path.exists(filepath2) and not os.path.exists(filepath3):
                    os.makedirs(filepath)
                    os.makedirs(filepath2)
                    os.makedirs(filepath3)


                for z in range(imagez):
                    image = sub_images[z, :, :]
                    image = image.astype(np.float32)
                    image = np.clip(image, 0, 255).astype('uint8')
                    cv2.imwrite(filepath + str(z) + ".bmp", image)
                    cv2.imwrite(filepath2 + str(z) + ".bmp", sub_masks[z, :, :])
                    cv2.imwrite(filepath3 + str(z) + ".bmp", sub_math[z, :, :])


def prepare3dtraindata(srcpath, maskpath, mathpath, trainImage, trainMask, trainMath, number, height, width, shape=(16, 256, 256),
                       numberxy=3, numberz=20):
    for i in range(0, number):
        index = 0
        listsrc = []
        listmask = []
        listmath = []
        break_flag = False
        file_list = os.listdir(srcpath + str(i))


        file_list = os.listdir(os.path.join(srcpath, str(i)))


        sorted_file_list = sorted(file_list, key=sort_key)

        depth_extracted = False
        for index, file_name in enumerate(sorted_file_list):
            image_path = os.path.join(srcpath, str(i), file_name)
            label_path = os.path.join(maskpath, str(i), file_name)
            math_path = os.path.join(mathpath, str(i), file_name)

            if os.path.exists(image_path) and os.path.exists(label_path) and os.path.exists(math_path):
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                math_image = cv2.imread(math_path, cv2.IMREAD_GRAYSCALE)

                if label.shape[0] != 512 or label.shape[1] != 512:
                    break_flag = True
                if not depth_extracted:
                    depth = extract_depth(file_name)
                    print("Depth:", depth)
                    depth_extracted = True

                listsrc.append(image)
                listmask.append(label)
                listmath.append(math_image)

        print(np.array(listsrc).shape)
        if break_flag == True:
            break_flag = False
            continue
        print("Number of images:", len(listsrc))
        print("Number of labels:", len(listmask))

        imagearray = np.array(listsrc)
        if (len(listsrc) == 0):
            continue
        imagearray = np.reshape(imagearray, (index + 1, height, width))
        maskarray = np.array(listmask)
        maskarray = np.reshape(maskarray, (index + 1, height, width))
        matharray = np.array(listmath)  
        matharray = np.reshape(matharray, (index + 1, height, width))

        gen_image_mask(imagearray, maskarray, matharray, i, shape=shape, numberxy=numberxy, numberz=numberz,
                       trainImage=trainImage, trainMask=trainMask, trainMath=trainMath, depth=depth)

def preparenoduledetectiontraindata():
    height = 512
    width = 512
    number = 999
    srcpath = r"D:\data_seg\process\image\\"
    maskpath = r"D:\data_seg\process\mask\\"
    mathpath = r"D:\data_seg\process\math\\"
    trainImage = r"D:\data_seg\segmentation\image\\"
    trainMask = r"D:\data_seg\segmentation\mask\\"
    trainMath = r"D:\data_seg\segmentation\math\\"
    prepare3dtraindata(srcpath, maskpath, mathpath, trainImage, trainMask, trainMath, number, height, width,
                       (32, 64, 64), 16, 8)

preparenoduledetectiontraindata()