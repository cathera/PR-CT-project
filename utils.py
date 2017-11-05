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
    info_list=[]
    i=-1
    for line in reader:
        if i==-1:
            i=i+1
            continue
        patient_path = './train_set/' + line[0] + '.mhd'
        info=func(patient_path)
        info['name']=line[0]
        info['z_index']=int((float(line[3])-info['origin'][2])/info['spacing'][2])
        info['x_location']=float(line[1])
        info['y_location']=float(line[2])
        info['diameter']=float(line[4])
        info_list.append(info)
        i=i+1
    return info_list

def get_info(patient_path):
    info={}
    img = sitk.ReadImage(patient_path)
    info['img'] = sitk.GetArrayFromImage(img)
    info['origin'] = np.array(img.GetOrigin())
    info['spacing'] = np.array(img.GetSpacing())
    return info

def resample(im_info, scale_r=0.5):
    # input im_info which is defined in func 'iter_samples'
    # output the resampled info
    info={}
    scale=im_info['spacing'][1]
    xy=np.arange(0, 512*scale, scale)
    xy_r=np.arange(0, 512*scale, scale_r)
    img=im_info['img'][im_info['z_index']]
    f=interp2d(xy,xy,img)
    info['img']=get_segmented_lungs(f(xy_r,xy_r))
    x_min=int((im_info['x_location']-im_info['origin'][0]-im_info['diameter']/2)/scale_r)
    x_center=int((im_info['x_location']-im_info['origin'][0])/scale_r)
    x_max=int((im_info['x_location']-im_info['origin'][0]+im_info['diameter']/2)/scale_r)
    y_min=int((im_info['y_location']-im_info['origin'][1]-im_info['diameter']/2)/scale_r)
    y_center=int((im_info['y_location']-im_info['origin'][1])/scale_r)
    y_max=int((im_info['y_location']-im_info['origin'][1]+im_info['diameter']/2)/scale_r)
    info['x']=[x_min,x_center,x_max]
    info['y']=[y_min,y_center,y_max]
    return info


def get_pn_sample(info, window_size=40):
    # To get the positive and negative samples in an img
    # info is defined in func 'resample'
img=info['img']
x_min=info['x'][0]
x_center=info['x'][1]
x_max=info['x'][2]
y_min=info['y'][0]
y_center=info['y'][1]
y_max=info['y'][2]
shape=img.shape[0]
samples={}
    '''Positive samples'''
    # 这里没考虑窗口取到黑区的情况
    center=img[int(y_center-window_size/2):int(y_center+window_size/2),int(x_center-window_size/2):int(x_center+window_size/2)]
    left=img[int(y_center-window_size/2):int(y_center+window_size/2),x_min:x_min+window_size]
    right=img[int(y_center-window_size/2):int(y_center+window_size/2),x_max-window_size:x_max]
    up=img[y_max-window_size:y_max,int(x_center-window_size/2):int(x_center+window_size/2)]
    down=img[y_min:y_min+window_size,int(x_center-window_size/2):int(x_center+window_size/2)]
    positive=[center,up,down,left,right]
    '''Negative samples'''
    negative=[]
    while len(negative)<5:
        x=np.random.randint(shape*0.2,0.8*shape-window_size)
        y=np.random.randint(shape*0.2,0.8*shape-window_size)
        mat=img[x:x+window_size,y:y+window_size]
        if (x>x_min & x<x_max & y<y_max & y>y_min):
            continue
        else:
            zeros=np.where(mat==0)
            if len(zeros[0])<40:
                negative.append(mat)             
    samples['positive']=positive
    samples['negative']=negative

    return samples

