from jpegCodec import JPEGencode, JPEGdecode
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import utils

# Load images
path1 = 'lena_color_512.png'
path2 = 'baboon.png'
images = (Image.open(path1), Image.open(path2))

subimg = [4, 4, 4]
qscale_values = (0.1, 0.3, 0.5, 0.6, 0.8, 1, 2, 5, 10)

for img, image in enumerate(images):

    images_rec = []
    errors = np.empty(len(qscale_values))
    bits = np.empty(len(qscale_values))

    for i, qscale in enumerate(qscale_values):

        # Encode image
        JPEGenc = JPEGencode(image, subimg, qscale, 0)
        meta = JPEGenc[0]
        bits[i] = meta.totalBits

        # Decoded image
        imgRec = JPEGdecode(JPEGenc)

        errors[i] = utils.mse(image, imgRec)
        images_rec.append(imgRec)
        print(errors[i])

    if img == 0: img_name = 'lena'
    elif img ==1: img_name = 'bbn'

    # Show reconstructed images
    plt.figure()
    plt.subplot(2, 5, 1)
    plt.imshow(image)
    plt.title('Original Image')
    for i, img in enumerate(images_rec):
        plt.subplot(2, 5, i+2)
        plt.imshow(img)
        plt.title(f'qscale = {qscale_values[i]}')
    plt.show()

    # MSE Plot
    values = ('0.1', '0.3', '0.5', '0.6', '0.8', '1', '2', '5', '10')
    plt.figure()
    plt.bar(values, errors, width = 0.6)
    plt.title(f'MSE for each quality value ({img_name})')
    plt.show()

    # Bit Plot
    plt.figure()
    plt.bar(values, bits, width = 0.6, color = 'orange')
    plt.title(f'Bit number for each quality value ({img_name})')
    plt.show()


