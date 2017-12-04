import csv
import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from scipy.interpolate import interp2d
from scipy.ndimage import zoom
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


def iter_samples():
    reader = csv.reader(open('./annotations.csv', encoding='utf-8'))
    t = ''
    for line in reader:
        if line[0] != t:
            if 'info' in locals():
                info['coords'] = np.array(coords).astype(int)
                info['diams'] = np.array(diams)[np.newaxis].T
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


def extract(im_info, scale_r=0.5, r=20):
    # for z in range(len(im_info['img'])):
    #     im_info['img'][z]=get_segmented_lungs(im_info['img'][z])
    samples = get_pn_samples(im_info, scale_r, r)
    for i, s in enumerate(samples):
        np.save('./samples/' + im_info['name'] + '_' + str(i), s)


def get_pn_samples(info, scale_r, r=20):
    # To get the positive and negative samples in an img
    # info is defined in func 'resample'
    r_x, r_y, r_z = np.ceil(info['spacing'] ** -1 * scale_r * r).astype(int)
    # bias = 15*scale_  # info['diams'] / 2  bias should be randomly chosen
    # X_window = np.repeat(np.array([-r_x,
    #     -bias, -2 * r_x + bias, -r_x, -r_x, -r_x, -r_x]), len(info['coords']))
    # Y_window = np.repeat(
    #     np.array([-r_y, -r_y, -r_y, -2 * r_y + bias, -bias, r_y, r_y]), len(info['coords']))
    # Z_window = np.repeat(np.array(
    #     [-r_z, -r_z, -r_z, -r_z, -r_z, -2 * r_z + bias, -bias]), len(info['coords']))
    # X = (np.tile(info['coords'][:, 0],
    #                 (1, 7)) + X_window).T.flatten().astype(int)
    # Y = (np.tile(info['coords'][:, 1],
    #                 (1, 7)) + Y_window).T.flatten().astype(int)
    # Z = (np.tile(info['coords'][:, 2],
    #                 (1, 7)) + Z_window).T.flatten().astype(int)
    # return X,Y,Z
    # positive = []
    # for x, y, z in info['coords']:
    #     img = zoom(info['img'][max(0, z - r_z):max(z + r_z, 2 * r_z), max(0, y - r_y):max(y + r_y, 2 * r_y),
    #                            max(0, x - r_x):max(x + r_x, 2 * r_x)], (r / r_z, r / r_y, r / r_x), order=3, mode='nearest')
    #     positive.extend((img, img.transpose((0, 2, 1)), img.transpose((1, 0, 2)), img.transpose(
    #         (1, 2, 0)), img.transpose((2, 1, 0)), img.transpose((2, 0, 1,))))
    #     positive.extend((img[:, :, ::-1], img[::-1, :, :], img[:, ::-1, :]))
    # return positive
    '''Negative samples'''
    shape = info['img'].shape
    nLen = len(info['coords'])*50
    negative=[]
    nX = np.random.randint(shape[1]*0.1,0.9*shape[1]-2*r_x, size=nLen)
    nY = np.random.randint(shape[2]*0.1,0.9*shape[2]-2*r_y, size=nLen)
    nZ = np.random.randint(shape[0]*0.1,0.9*shape[0]-2*r_z, size=nLen)
    i=0
    while len(negative)< nLen/5 and i<nLen:
        if coords_range(info['coords'],[nZ[i],nX[i],nY[i]], r_x):
            mat=info['img'][nZ[i]:nZ[i]+2*r_z,nX[i]:nX[i]+2*r_x,nY[i]:nY[i]+2*r_y]
            zeros=np.where(mat>=50)
            blacks=np.where(mat<-1000)
            if len(zeros[0])<40*40*15 & len(blacks[0])<40*40*15 :
                negative.append(zoom(mat, (r / r_z, r / r_y, r / r_x), order=3, mode='nearest'))
            else:
                if (len(zeros[0])>40*40*6):
                    print('border zero')
                else:
                    print('border blacks')
            i=i+1
    
        else:
            print('to close')
            i=i+1
            continue
    return negative#positive,negative


def coords_range(coords, test, r=20):
    distance = abs(coords - test)
    for v in distance:
        if len(np.where(v < 2 * r)[0]) >= 3:
            return False
    return True

def preprocess(sample, spacing_r=1, scale=512):
    # Segment the lung
    # Resample a sample by 1*1*1 spacing
    # Then cut it to size 512*512
    def segment_resample_resize(img):
        # img is a 2d matirx
        '''segment'''
        img = get_segmented_lungs(img)
        '''resample'''
        xy = np.arange(0, 512 * sample['spacing'][0], sample['spacing'][0])
        xy_r = np.arange(0, 512 * sample['spacing'][0], spacing_r)
        f = interp2d(xy, xy, img)
        img = f(xy_r, xy_r)
        '''resize'''
        return img
    sample['img']=list(map(segment_resample_resize,sample['img']))
    return sample