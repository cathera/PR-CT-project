from utils import get_segmented_lungs
import numpy as np




def pre_process(sample, scale=512):
    # Segment the lung
    # Resample a sample by 1*1*1 spacing
    # Then cut it to size 512*512
    def segment_resample_resize(img, spacing_r=1, size=512):
        # img is a 2d matirx
        '''segment'''
        img = get_segmented_lungs(img)
        '''resample'''
        xy = np.arange(0, 512 * sample['spacing'][0], spacing[0])
        xy_r = np.arange(0, 512 * sample['spacing'][0], spacing_r)
        f = interp2d(xy, xy, img)
        img_r = f(xy_r, xy_r)
        '''resize'''
        return img_r
    img=list(map(segment_resample_resize,sample['img']))
    return sample
    