from CAPTCHAgenerator import CAPTCHAgenerator
from matplotlib import pyplot as plt
from Preprocess import Preprocess

a = CAPTCHAgenerator(1)
b = Preprocess(a.get_path())
images = b.preprocess()
b.save_imgs()
# for i in range(len(images)):
    # plt.subplot(10, 4, i + 1)
    # print(images[i].shape)
    # plt.imshow(images[i], cmap='gray')

plt.show()
# pyplot.imshow(a.get_CAPTCHA(), cmap=pyplot.get_cmap('gray'))
# pyplot.show()
