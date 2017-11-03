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

def get_segmented_lungs(im, plot=False, THRESHOLD = -320):

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
        plots[0,0].axis('off')
        plots[0,0].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1,0].axis('off')
        plots[1,0].imshow(cleared, cmap=plt.cm.bone)
    '''
    Step 3: Label the image.
    '''
    label_image = measure.label(cleared)
    if plot == True:
        plots[2,0].axis('off')
        plots[2,0].imshow(label_image, cmap=plt.cm.bone)
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in measure.regionprops(label_image)]
    areas.sort()
    #print areas
    if len(areas) > 2:
        for region in measure.regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[0,1].axis('off')
        plots[0,1].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[1,1].axis('off')
        plots[1,1].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is
    to keep nodules attached to the lung wall.
    '''
    selem = disk(15)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[2,1].axis('off')
        plots[2,1].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[0,2].axis('off')
        plots[0,2].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot == True:
        plots[1,2].axis('off')
        plots[1,2].imshow(im, cmap=plt.cm.bone)

    return im

def iter_samples(func):
    reader = csv.reader(open('./annotations.csv', encoding='utf-8'))
    for line in reader:
        patient_path = './train_set/' + line[0] + '.mhd'
        func(patient_path)

def get_info(patient_path):
    info={}
    img = sitk.ReadImage(patient_path)
    info['img'] = sitk.GetArrayFromImage(img)
    info['origin'] = np.array(img.GetOrigin())
    info['spacing'] = np.array(img.GetSpacing())
    return info

def resample(im_info, z, scale_r=0.5):
    # z should be a list of required z_locations
    # im_info comes from get_info
    # Sample usage: y=resample(get_info('./train_set/LKDS-00001.mhd'), [100,152])
    scale=im_info['spacing'][1]
    xy=np.arange(0, 512*scale, scale)
    xy_r=np.arange(0, 512*scale, scale_r)
    imgs=[]
    for z_ in z:
        img=im_info['img'][z_]
        f=interp2d(xy,xy,img)
        imgs.append(f(xy_r,xy_r))
    return imgs
