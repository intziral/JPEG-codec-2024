from jpegCodec import JPEGencode, JPEGdecode
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

def spatial_entropy(image):
    rgb_values = np.array(list(image.getdata()))
    # count unique RGB values
    _, counts = np.unique(rgb_values, axis=0, return_counts=True)
    probabilities = counts / (image.size[0] * image.size[1])

    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# Load images
path1 = 'lena_color_512.png'
path2 = 'baboon.png'
images = (Image.open(path1), Image.open(path2))

subimg = ([4, 2, 0], [4, 4, 4])
qscale = (0.6, 5)

for i, image in enumerate(images):

    print(f'\nImage {i+1} ...')

    # Entropy of original image
    entropy = spatial_entropy(image)
    print('-Spatial entropy =', round(entropy, 4))

    # Encode image
    JPEGenc = JPEGencode(image, subimg[i], qscale[i], 0)

    # Reconstructed image
    imgRec = JPEGdecode(JPEGenc)

    if i == 0: img_name = 'lena'
    elif i == 1: img_name = 'bbn'
    
    # Show reconstructed image
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(imgRec)
    plt.title('Reconstructed Image')
    plt.show()