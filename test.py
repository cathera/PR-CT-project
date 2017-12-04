import utils

sample_list = utils.iter_samples()
sample = next(sample_list)
img = sample['img']
img_r=utils.segment_resample_resize(img[100],sample['spacing'])

for i in [1,20,40,60]:
    img=s['img'][i]
    lung=utils.get_segmented_lungs(img)
    lung_area=utils.np.where(lung!=0)
    lung_area=lung_area[0].shape[0]
    print(i,lung_area/(512*512))
    utils.plt.figure()
    utils.plt.imshow(lung)

utils.plt.show()
