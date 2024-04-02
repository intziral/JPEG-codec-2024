from PIL import Image
from matplotlib import pyplot as plt
import utils
import numpy as np

# JPEG Library
import jpegLibrary as j

# Load images
path1 = 'lena_color_512.png'
path2 = 'baboon.png'
images = (Image.open(path1), Image.open(path2))

subsampling = ([4, 2, 2], [4, 4, 4])

# Part 1:
print("\nExample 1 ...")

for im, image in enumerate(images):

    print("\nImage", im+1)

    # Convert rgb image to Y, Cr, Cb channels
    imageY, imageCr, imageCb = j.convert2ycrcb(image, subsampling[im])

    print('-No quantization')

    # Convert back to rgb
    imageRGB = j.convert2rgb(imageY, imageCr, imageCb, subsampling[im])

    # Show reconstructed image
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(images[im])
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(imageRGB)
    plt.title('Reconstructed Image')
    plt.show()

# Part 2:
print("\nExample 2 ...")
qscale = (0.6, 5)

for im, image in enumerate(images):
    
    print("\nImage", im+1)

    # Convert rgb image to Y, Cr, Cb channels
    imageY, imageCr, imageCb = j.convert2ycrcb(image, subsampling[im])

    # Split images to 8x8 blocks
    blocksY = utils.to_blocks(imageY)
    blocksCr = utils.to_blocks(imageCr)
    blocksCb = utils.to_blocks(imageCb)

    # Initialize DCT arrays
    dctBlocksY = np.zeros((len(blocksY), 8, 8))
    dctBlocksCr = np.zeros((len(blocksCr), 8, 8))
    dctBlocksCb = np.zeros((len(blocksCb), 8, 8))

    # DCT for Luminence blocks
    for i in range(len(blocksY)):
        dctBlocksY[i] = j.blockDCT(blocksY[i])

    # DCT for Chrominance blocks
    for i in range(len(blocksCr)):
        dctBlocksCr[i] = j.blockDCT(blocksCr[i])
        dctBlocksCb[i] = j.blockDCT(blocksCb[i])

    print("-Quantization with qscale =", qscale[im])

    # Initialize Quantization arrays
    qBlocksY = np.zeros((len(blocksY), 8, 8))
    qBlocksCr = np.zeros((len(blocksCb), 8, 8))
    qBlocksCb = np.zeros((len(blocksCb), 8, 8))
    qTableL = utils.Q_y
    qTableC = utils.Q_c

    # Quantize Luminence blocks
    for i in range(len(blocksY)):
        qBlocksY[i] = j.quantizeJPEG(dctBlocksY[i], qTableL, qscale[im])

    # Quantize Chrominance blocks
    for i in range(len(blocksCr)):
        qBlocksCr[i] = j.quantizeJPEG(dctBlocksCr[i], qTableC, qscale[im])
        qBlocksCb[i] = j.quantizeJPEG(dctBlocksCb[i], qTableC, qscale[im])

    # Dequantize Luminence blocks
    for i in range(len(blocksY)):
        dctBlocksY[i] = j.dequantizeJPEG(qBlocksY[i], qTableL, qscale[im])

    # Dequantize Chrominance blocks
    for i in range(len(blocksCr)):
        dctBlocksCr[i] = j.dequantizeJPEG(qBlocksCr[i], qTableC, qscale[im])
        dctBlocksCb[i] = j.dequantizeJPEG(qBlocksCb[i], qTableC, qscale[im])
    
    # Initialize arrays for Inverse DCT
    idctBlocksY = np.zeros((len(blocksY), 8, 8))
    idctBlocksCr = np.zeros((len(blocksCr), 8, 8))
    idctBlocksCb = np.zeros((len(blocksCb), 8, 8))

    # iDCT:
    for i in range(len(blocksY)):
        idctBlocksY[i] = j.iBlockDCT(dctBlocksY[i])
    for i in range(len(blocksCr)):
        idctBlocksCr[i] = j.iBlockDCT(dctBlocksCr[i])
        idctBlocksCb[i] = j.iBlockDCT(dctBlocksCb[i])
    
    # Blocks back to image shapes
    imageY = utils.from_blocks(idctBlocksY, imageY.shape)
    imageCr = utils.from_blocks(idctBlocksCr, imageCr.shape)
    imageCb = utils.from_blocks(idctBlocksCb, imageCb.shape)

    # Convert back to rgb
    imageRGB = j.convert2rgb(imageY, imageCr, imageCb, subsampling[im])

    # Show reconstructed image
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(images[im])
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(imageRGB)
    plt.title('Reconstructed Image')
    plt.show()

    