# File containing the functions for JPEG encoding and decoding
import numpy as np
import cv2
from scipy.fftpack import dct, idct
import utils

# Convert RGB images to Y, Cr, Cb tables
def convert2ycrcb(imageRGB, subimg):

    imageRGB = np.array(imageRGB)
    
    # Get the dimensions of the image
    original_height, original_width, _ = imageRGB.shape
    print("-Image original size:", imageRGB.shape)

    # Crop image to height and width divisible by 8
    height = original_height - (original_height % 8)
    width = original_width - (original_width % 8)
    if height != original_height:
        imageRGB = imageRGB[0:height, 0:width]
        print("-Image was cropped to size:", imageRGB.shape)

    # YCrCb Transformation
    conversion_matrix = np.array([[.299, .587, .114], # Conversion matrix from RGB to YCrCB
                                [.5, -.4187, -.0813],
                                [-.1687, -.3313, .5],
                                ])
    imageYCrCb = imageRGB.dot(conversion_matrix.T)
    imageYCrCb[:, :, [1, 2]] += 128

    # Assign values to imageY, imageCr, and imageCb
    imageY = imageYCrCb[:, :, 0]
    imageCr_values = imageYCrCb[:, :, 1]    
    imageCb_values = imageYCrCb[:, :, 2]

    # Chrominance subsampling:
    if subimg == [4, 4, 4]:
        # No subsampling
        imageCr = imageCr_values
        imageCb = imageCb_values
    
    elif subimg == [4, 2, 2]:
        # Skip every second colum
        imageCr = imageCr_values[:, 0::2]
        imageCb = imageCb_values[:, 0::2]
    
    elif subimg == [4, 2, 0]:
        # Skip every second pixel
        imageCr = imageCr_values[0::2, 0::2]
        imageCb = imageCb_values[0::2, 0::2]

    # Return the separate Y, Cr, and Cb integer values
    return np.uint8(imageY), np.uint8(imageCr), np.uint8(imageCb)

# Convert Y, Cr, Cb values back to RGB image
def convert2rgb(imageY, imageCr, imageCb, subimg):

    # Get original size
    height, width = imageY.shape
    imageY_values = imageY

    # Chrominance upsampling
    if subimg == [4, 4, 4]:
        # No upsampling needed
        imageCr_values = imageCr
        imageCb_values = imageCb
    
    elif subimg == [4, 2, 2] or subimg == [4, 2, 0]:
        # Upsamling to original image size
        imageCr_values = cv2.resize(imageCr, dsize=(width, height))
        imageCb_values = cv2.resize(imageCb, dsize=(width, height))
    
    # RGB Transformation
    imageYCrCb = np.zeros((height, width, 3))
    imageYCrCb[:, :, 0] = imageY_values
    imageYCrCb[:, :, 1] = imageCr_values
    imageYCrCb[:, :, 2] = imageCb_values
    imageYCrCb[:, :, [1, 2]] -= 128
    inverse_conversion_matrix = np.array([[1, 1.4102, 0],   # Conversion matrix from RGB to YCrCB
                                        [1, -.7141, -.3441],
                                        [1, 0, 1.772],
                                        ])
    imageRGB = imageYCrCb.dot(inverse_conversion_matrix.T)  # Dot product

    # Adjust values to [0, 255] range
    np.putmask(imageRGB, imageRGB > 255, 255)
    np.putmask(imageRGB, imageRGB < 0, 0)

    # Return integer values
    return np.uint8(imageRGB)


# DCT using the dct scipy function
def blockDCT(block):
    
    dctBlock = dct(block, type=2, norm='ortho')

    return dctBlock

# Inverse DCT using the idct scipy function
def iBlockDCT(dctBlock):

    block = idct(dctBlock, type=2, norm='ortho')
    return block


# Quantization function
def quantizeJPEG(dctBlock, qtable, qscale):

    # Quantization
    qblock = np.round(dctBlock / (qtable * qscale))

    return qblock

# Dequantization function
def dequantizeJPEG(qblock, qtable, qscale):
    
    # Dequantization
    dctBlock = qblock * qtable * qscale
    
    return dctBlock


# Run length encoding function
def runLength(qblock, DCpred):

    diff = qblock[0, 0] - DCpred # DIFF = DC - PRED

    zz_sequence = utils.zig_zag(qblock) # zig-zag function

    # Append DC component
    runSymbols = [diff]

    # RLE for AC components
    i = 0
    while (i <= len(zz_sequence) - 1):
        count = 0
        j = i
        while (j < len(zz_sequence) - 1):
            if (zz_sequence[j] == 0):
                count += 1
                j += 1
            else:
                break
        # Append length, value pairs
        runSymbols.append((count, zz_sequence[j]))
        i = j + 1

    return runSymbols

# Run length decoding function
def irunLength(runSymbols, DCpred):

    # Init sequence
    DC = runSymbols[0] + DCpred # DC = DIFF + PRED
    sequence = [DC]
    n = runSymbols[1:]  # symbols

    # iRLE
    k = 0
    for i in range(len(n)):
        # EOB
        if (n[i] == (0, 0)):
            while (k < 63):
                sequence.append(0)
                k += 1
            break
        # other symbols
        else:
            count, symbol = n[i]
            j = 0
            while (j < count):
                sequence.append(0)
                j += 1
            sequence.append(symbol)
            k += count + 1

    # Build block from zig_zag sequence
    qblock = utils.izig_zag(sequence, (8, 8))  
    
    return qblock


# Huffman encoding function
def huffEnc(runSymbols, huffTables):
    
    # Initialize Huffman stream
    huffStream = ''

    # Huffman tables
    huffDC, huffAC = huffTables

    # Encode DC component
    diff = runSymbols[0]
    if (diff == 0):
        huffStream += '00' # no extra bits
    else:
        ssss, code_index = utils.index_2d(huffCategories, diff) # size category
        xtra_bits = '{:0{padding}b}'.format(code_index, padding=ssss) # extra bits
        # DC code word from Huffman table
        DC_code_word = huffDC[ssss] + xtra_bits
        huffStream += DC_code_word

    # Encode AC components
    AC_components = runSymbols[1:]
    for pair in AC_components:
        length, value = pair
        if (value == 0):
            (rrrr, ssss) = (0, 0)   # code EOB
            huffStream += huffAC[(rrrr, ssss)] 
        elif (length > 15):
            k = length
            while (k > 15):
                huffStream += huffAC[(15, 0)] # code ZRL
                k -= 16
            rrrr = k
            ssss, code_index = utils.index_2d(huffCategories, value) # size category
            # AC code word from Huffman table
            xtra_bits = '{:0{padding}b}'.format(code_index, padding=ssss)
            AC_code_word = huffAC[(rrrr, ssss)] + xtra_bits
            huffStream += AC_code_word
        else:
            rrrr = length
            ssss, code_index = utils.index_2d(huffCategories, value) # size category
            # AC code word from Huffman table
            xtra_bits = '{:0{padding}b}'.format(code_index, padding=ssss)
            AC_code_word = huffAC[(rrrr, ssss)] + xtra_bits
            huffStream += AC_code_word
    
    return huffStream

def huffDec(huffStream, huffTables):

    def decHuffCode(bitstream_iter, huffTable):
        code = ''
        for bit in bitstream_iter:
            code += bit
            if code in huffTable.values():
                return list(huffTable.keys())[list(huffTable.values()).index(code)]
            
    def decXtraBits(bitstream_iter, ssss):
        xtra_bits = ''
        for bit in bitstream_iter:
            xtra_bits += bit
            if (len(xtra_bits) == ssss):
                return huffCategories[ssss][int(xtra_bits, 2)]

    huffDC, huffAC = huffTables
    huffStream_iter = iter(huffStream)
    runSymbols = []

    # Decode DC component
    ssss = decHuffCode(huffStream_iter, huffDC)
    if ssss == 0:
        DC_symbol = 0
    else:
        DC_symbol = decXtraBits(huffStream_iter, ssss)

    runSymbols.append(DC_symbol)

    # Decode AC component
    k = 0
    while True:
        rrrr, ssss = decHuffCode(huffStream_iter, huffAC)
        # ZRL
        if ((rrrr, ssss) == (15, 0)):
            runSymbols.append((15, 0))
            k += 16
        # EOB
        elif (ssss == 0):
            runSymbols.append((0, 0))
            break
        else:
            AC_symbol = decXtraBits(huffStream_iter, ssss)
            runSymbols.append((rrrr, AC_symbol))
            k += rrrr + 1
        if (k == 63): break

    return runSymbols

huffCategories = (
    (0, ),
    (-1, 1),
    (-3, -2, 2, 3),
    (*range(-7, -4 + 1), *range(4, 7 + 1)),
    (*range(-15, -8 + 1), *range(8, 15 + 1)),
    (*range(-31, -16 + 1), *range(16, 31 + 1)),
    (*range(-63, -32 + 1), *range(32, 63 + 1)),
    (*range(-127, -64 + 1), *range(64, 127 + 1)),
    (*range(-255, -128 + 1), *range(128, 255 + 1)),
    (*range(-511, -256 + 1), *range(256, 511 + 1)),
    (*range(-1023, -512 + 1), *range(512, 1023 + 1)),
    (*range(-2047, -1024 + 1), *range(1024, 2047 + 1)),
)