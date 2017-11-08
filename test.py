import utils

l = utils.iter_samples()
s = next(l)
s = next(l)

img = s['img'][s['coords'][0][0]]
pn = utils.get_pn_samples(s, 1)
Len = len(pn)
print(Len)
utils.plt.imshow(img)
utils.plt.show()