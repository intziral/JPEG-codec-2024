from jpegCodec import JPEGencode, JPEGdecode
from PIL import Image
from matplotlib import pyplot as plt

# Load images
path1 = 'lena_color_512.png'
path2 = 'baboon.png'
images = (Image.open(path1), Image.open(path2))

subimg = [4, 4, 4]
zero_numbers = (20, 40, 50, 60, 63)
qscale = 1

for img, image in enumerate(images):

    images_rec = []

    for i, zeros in enumerate(zero_numbers):

        # Encode image
        JPEGenc = JPEGencode(image, subimg, qscale, zeros)

        # Decoded image
        imgRec = JPEGdecode(JPEGenc)
        images_rec.append(imgRec)

    if img == 0: img_name = 'lena'
    elif img ==1: img_name = 'bbn'

    # Show reconstructed images
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    for i, img in enumerate(images_rec):
        plt.subplot(2, 3, i+2)
        plt.imshow(img)
        plt.title(f'zeros = {zero_numbers[i]}')
    plt.show()

    plt.figure()
    plt.imshow(images_rec[1])
    plt.show

