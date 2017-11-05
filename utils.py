import csv
import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from scipy.interpolate import interp2d
from skimage import measure
from skimage.segmentation import clear_border
from skimage.morphology import disk, binary_erosion, binary_closing
from skimage.filters import roberts


def get_segmented_lungs(im, plot=False, THRESHOLD=-320):
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(3, 3, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image.
    '''
    binary = im < THRESHOLD
    if plot == True:
        plots[0, 0].axis('off')
        plots[0, 0].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1, 0].axis('off')
        plots[1, 0].imshow(cleared, cmap=plt.cm.bone)
    '''
    Step 3: Label the image.
    '''
    label_image = measure.label(cleared)
    if plot == True:
        plots[2, 0].axis('off')
        plots[2, 0].imshow(label_image, cmap=plt.cm.bone)
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in measure.regionprops(label_image)]
    areas.sort()
    # print areas
    if len(areas) > 2:
        for region in measure.regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[0, 1].axis('off')
        plots[0, 1].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[1, 1].axis('off')
        plots[1, 1].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is
    to keep nodules attached to the lung wall.
    '''
    selem = disk(15)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[2, 1].axis('off')
        plots[2, 1].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[0, 2].axis('off')
        plots[0, 2].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot == True:
        plots[1, 2].axis('off')
        plots[1, 2].imshow(im, cmap=plt.cm.bone)
    return im


def iter_samples(func):
    reader = csv.reader(open('./annotations.csv', encoding='utf-8'))
    t = ''
    for line in reader:
        if line[0] != t:
            if 'info' in locals():
                info['coords'] = np.array(coords)
                info['diams'] = np.array(diams)[np.newaxis].T
                # extract(info)
                yield info
            info = get_info(line)
            coords = []
            diams = []
        coords.append((np.array([float(line[1]), float(line[2]), float(
            line[3])]) - info['origin']) / info['spacing'])
        diams.append(float(line[4]))
        t = line[0]


def get_info(line):
    info = {}
    img = sitk.ReadImage('./train_set/' + line[0] + '.mhd')
    info['name'] = line[0]
    info['img'] = sitk.GetArrayFromImage(img)
    info['origin'] = np.array(img.GetOrigin())
    info['spacing'] = np.array(img.GetSpacing())
    info['locations'] = []
    return info


def resample(im_info, z, spacing_r=0.5):
    # z is a list!
    # Dont mess with z axis, 3D resampling requires a rediculously long time
    img_r = []
    spacing = im_info['spacing']
    xy = np.arange(0, 512 * spacing[0], spacing[0])
    xy_r = np.arange(0, 512 * spacing[0], spacing_r[0])
    for z_ in z:
        img = im_info['img'][z_]
        f = interp2d(xy, xy, img)
        img_r.append(f(xy_r, xy_r))
    im_info['img'] = np.array(img_r)
    im_info['coords'][:, :2] *= spacing[0] / spacing_r
    im_info['spacing'] = np.array([spacing_r, spacing_r, spacing[2]])
    return im_info


def extract(im_info):
    pass


def get_pn_sample(info, window_size=40):
    # To get the positive and negative samples in an img
    # info is defined in func 'resample'
    r = window_size / 2
    z_pos = info['coords'][:, 2].astype(int)
    bias = 15  # info['diams'] / 2  bias should be randomly chosen
    coords_center = info['coords'][:, :2]
    X_window = np.tile(np.array([[-r, r], [-bias, 2 * r - bias],
                                 [-2 * r + bias, bias], [-r, r], [-r, r]]), len(z_pos)).reshape(-1, 2)
    Y_window = np.tile(np.array([[-r, r], [-r, r], [-r, r], [-2 * r +
                                                             bias, bias], [-bias, 2 * r - bias]]), len(z_pos)).reshape(-1, 2)
    X = (np.tile(coords_center[:, 0][np.newaxis],
                 (2, 5)).T + X_window).astype(int)
    Y = (np.tile(coords_center[:, 1][np.newaxis],
                 (2, 5)).T + Y_window).astype(int)
    Z = np.tile(z_pos, 5)
    '''Positive samples'''
    # 这里没考虑窗口取到黑区的情况
    positive = []
    for i in range(len(Z)):
        print(Z[i], Y[i], X[i])
        positive.append(info['img'][Z[i], Y[i][0]:Y[i][1], X[i][0]:X[i][1]])
    '''Negative samples'''
    # Above calculations are all done in matrix form
    # TODO: Adapt negative sampling
    # negative=[]
    # while len(negative)<5:
    #     x=np.random.randint(shape*0.2,0.8*shape-window_size)
    #     y=np.random.randint(shape*0.2,0.8*shape-window_size)
    #     mat=img[x:x+window_size,y:y+window_size]
    #     if (x>x_min & x<x_max & y<y_max & y>y_min):
    #         continue
    #     else:
    #         zeros=np.where(mat==0)
    #         if len(zeros[0])<40:
    #             negative.append(mat)
    # samples['positive']=positive
    # samples['negative']=negative

    return positive
