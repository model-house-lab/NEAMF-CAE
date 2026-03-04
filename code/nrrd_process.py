from __future__ import print_function, division
import os
import SimpleITK as sitk
import cv2
import pandas as pd
import re
import nrrd
import numpy as np


def mark_connected_region_3d(grid):
    m, n, p = len(grid), len(grid[0]), len(grid[0][0])
    connected_region_grid = [[[0] * p for _ in range(n)] for _ in range(m)]
    current_label = 255
    stack = []

    for i in range(m):
        for j in range(n):
            for k in range(p):
                if grid[i][j][k] == 255 and connected_region_grid[i][j][k] == 0:
                    current_label -= 1
                    print("current_label=", current_label)
                    if current_label % 10 == 0:
                        print("current_label=", current_label)
                    if current_label == 0:
                        break;
                    stack.append((i, j, k))
                    while stack:
                        x, y, z = stack.pop()
                        if x < 0 or x >= m or y < 0 or y >= n or z < 0 or z >= p or grid[x][y][z] == 0 or \
                                connected_region_grid[x][y][z] != 0:
                            continue
                        connected_region_grid[x][y][z] = current_label
                        stack.extend(
                            [(x + 1, y, z), (x - 1, y, z), (x, y + 1, z), (x, y - 1, z), (x, y, z + 1), (x, y, z - 1)])

    return connected_region_grid, current_label


def getRangImageDepth(image):
    fistflag = True
    startposition = 0
    endposition = 0
    for z in range(image.shape[2]):
        notzeroflag = np.max(image[:, :, z])
        if notzeroflag and fistflag:
            startposition = z
            fistflag = False
        if notzeroflag:
            endposition = z
    return startposition, endposition


def sort_key(name):
    digits = re.findall(r'\d+', name)
    if digits:
        return int(digits[0])
    else:
        return float('inf')


def resize_image_itk(itkimage, newSpacing, resamplemethod=sitk.sitkNearestNeighbor):

    newSpacing = np.array(newSpacing, float)
    originSpcaing = itkimage.GetSpacing()
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    factor = newSpacing / originSpcaing
    newSize = originSize / factor
    newSize = newSize.astype(np.int)
    resampler.SetReferenceImage(itkimage)
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetSize(newSize.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    if resamplemethod == sitk.sitkNearestNeighbor:
        itkimgResampled = sitk.Threshold(itkimgResampled, 0, 1.0, 255)
    imgResampled = sitk.GetArrayFromImage(itkimgResampled)
    return imgResampled, itkimgResampled


def load_itk(filename):

    rescalFilt = sitk.RescaleIntensityImageFilter()
    rescalFilt.SetOutputMaximum(255)
    rescalFilt.SetOutputMinimum(0)
    itkimage = rescalFilt.Execute(sitk.Cast(sitk.ReadImage(filename), sitk.sitkFloat32))
    return itkimage


def load_itkfilewithtrucation(filename, upper=200, lower=-200):

    srcitkimage = sitk.Cast(sitk.ReadImage(filename), sitk.sitkFloat32)
    srcitkimagearray = sitk.GetArrayFromImage(srcitkimage)
    srcitkimagearray[srcitkimagearray > upper] = upper
    srcitkimagearray[srcitkimagearray < lower] = lower
    sitktructedimage = sitk.GetImageFromArray(srcitkimagearray)
    origin = np.array(srcitkimage.GetOrigin())
    spacing = np.array(srcitkimage.GetSpacing())
    sitktructedimage.SetSpacing(spacing)
    sitktructedimage.SetOrigin(origin)
    rescalFilt = sitk.RescaleIntensityImageFilter()
    rescalFilt.SetOutputMaximum(255)
    rescalFilt.SetOutputMinimum(0)
    itkimage = rescalFilt.Execute(sitk.Cast(sitktructedimage, sitk.sitkFloat32))
    return itkimage


def read_tumor_labels(folder_path):

    labels_dict = {}
    for filename in sorted(os.listdir(folder_path), key=sort_key):
        if filename.startswith("label"):
            parts = filename.split("_")
            sample_id = int(parts[0][6:])
            tumor_label = os.path.splitext(filename)[0]
            if sample_id not in labels_dict:
                labels_dict[sample_id] = []
            labels_dict[sample_id].append(tumor_label)
    labels_array = []
    for sample_id, tumor_labels in sorted(labels_dict.items()):
        tumor_labels.extend([""] * (3 - len(tumor_labels)))
        labels_array.append(tumor_labels[:3])
    return labels_array


def processOriginaltraindata():
    expandslice = 32
    trainImage = r"D:\data_seg\process\image\\"
    trainMask = r"D:\data_seg\process\mask\\"
    trainMath = r"D:\data_seg\process\math\\"

    """
    load itk image,change z Spacing value to 1,and save image ,liver mask ,tumor mask
    :return:None
    """
    seriesindex = 0
    seriesindex_1 = 0
    for subsetindex in range(1):
        luna_paths_df = pd.read_csv(r"D:\data_seg\csv\image_test.csv")
        luna_paths = luna_paths_df['File Path'].tolist()
        luna_paths = sorted(luna_paths, key=sort_key)
        output_path = read_tumor_labels(r"D:\data_seg\mask\\")
        luna_subset_path = luna_paths
        luna_subset_mask_path = output_path
        file_list = luna_paths
        for fcount in range(0, len(file_list)):
            print("fcount=", fcount)
            src, _ = nrrd.read(file_list[fcount])
            print("np.max==", np.max(src))
            src[src > 600] = 600
            src[src < -1000] = -1000
            src = (src + 1000) * (255.0 / 1600.0)
            print("np.max==", np.max(src))
            print("np.min==", np.min(src))
            print(file_list[fcount])
            print(src.shape)
            sub_img_file = file_list[fcount][len(luna_subset_path):-4]
            seg = []

            # 读取对应的肿瘤标签
            for index in range(len(output_path[fcount])):
                print(r"D:\shenxi\SRY\mask_test\\\\" + output_path[fcount][index])
                if (output_path[fcount][index] == ""):
                    break
                seg_index, _ = nrrd.read(r"D:\data_seg\mask\\" + output_path[fcount][index] + ".nrrd")
                seg.append(seg_index)

            seg = np.maximum.reduce(seg)
            seg[seg > 0] = 255
            srcimg = src
            segimg = seg

            segimg, label = mark_connected_region_3d(segimg)

            srcimg = np.array(srcimg)
            srcimg = np.rot90(srcimg, k=3, axes=(0, 1))
            segimg = np.array(segimg)
            segimg = np.rot90(segimg, k=3, axes=(0, 1))

            # 读取对应的 math 数据
            math_img = []
            for index in range(len(output_path[fcount])):
                if (output_path[fcount][index] == ""):
                    break
                math_index, _ = nrrd.read(r"D:\data_seg\math\\" + output_path[fcount][index] + ".nrrd")
                math_img.append(math_index)


            math_img = np.maximum.reduce(math_img)
            math_img[math_img > 0] = 255
            math_liverimage = math_img
            math_liverimage = np.array(math_liverimage)
            math_liverimage = np.rot90(math_liverimage, k=3, axes=(0, 1))
            trainimagefile = trainImage + str(seriesindex_1)
            trainMaskfile = trainMask + str(seriesindex_1)
            trainMathfile = trainMath + str(seriesindex_1)

            if not os.path.exists(trainimagefile):
                os.makedirs(trainimagefile)
            if not os.path.exists(trainMaskfile):
                os.makedirs(trainMaskfile)
            if not os.path.exists(trainMathfile):
                os.makedirs(trainMathfile)

            seg_liverimage = segimg.copy()

            startpostion, endpostion = getRangImageDepth(seg_liverimage)
            if startpostion == endpostion:
                seriesindex_1 += 1
                continue
            imagez = np.shape(seg_liverimage)[2]
            startpostion = startpostion - expandslice
            endpostion = endpostion + expandslice
            if startpostion < 0:
                startpostion = 0
            if endpostion > imagez:
                endpostion = imagez

            srcimg = srcimg[:, :, startpostion:endpostion]
            seg_liverimage = seg_liverimage[:, :, startpostion:endpostion]
            math_liverimage = math_liverimage[:, :, startpostion:endpostion]

            threshold_value = 0

            seriesindex = 0
            for z in range(seg_liverimage.shape[2]):
                srcimg = np.clip(srcimg, 0, 255).astype('uint8')

                non_zero_pixels = np.count_nonzero(seg_liverimage[:, :, z])
                print(f"Slice {z}: Number of non-zero pixels: {non_zero_pixels}")

                if non_zero_pixels >= threshold_value:
                    file_name = f"{trainimagefile}\\{str(seriesindex)}_z{startpostion + z}.bmp"
                    cv2.imwrite(file_name, np.fliplr(srcimg[:, :, z]))

                    mask_file_name = f"{trainMaskfile}\\{str(seriesindex)}_z{startpostion + z}.bmp"
                    cv2.imwrite(mask_file_name, np.fliplr(seg_liverimage[:, :, z]))

                    math_file_name = f"{trainMathfile}\\{str(seriesindex)}_z{startpostion + z}.bmp"
                    cv2.imwrite(math_file_name, np.fliplr(math_liverimage[:, :, z]))

                    seriesindex += 1
                else:
                    print(f"Skipping slice {startpostion + z} due to insufficient non-zero pixels.")
            seriesindex_1 += 1

processOriginaltraindata()