import numpy as np
import utils
import math
from matplotlib import pyplot as plt

# JPEG Library
import jpegLibrary as j

class metadata:

    def __init__(self, qScale, qTableL, qTableC, DCL, DCC, ACL, ACC, totalBits):

        self.qScale = qScale
        self.qTableL = qTableL
        self.qTableC = qTableC
        self.DCL = DCL
        self.DCC = DCC
        self.ACL = ACL
        self.ACC = ACC
        self.totalBits = totalBits
    
class block:

    def __init__(self, blkType, indHor, indVer, huffStream):

        self.blkType = blkType
        self.indHor = indHor
        self.indVer = indVer
        self.huffStream = huffStream

    # Function to decode each block
    def decode(self, huffTables, DCpred, qTable, qScale):

        runSymbols = j.huffDec(self.huffStream, huffTables)     # Huffman decoding
        qBlock = j.irunLength(runSymbols, DCpred)               # iRLE
        DCpred = qBlock[0, 0]                                   # Set DC prediction for next block
        dctBlock = j.dequantizeJPEG(qBlock, qTable, qScale)     # Dequantization
        block = j.iBlockDCT(dctBlock)                           # iDCT

        return block, DCpred, self.indHor, self.indVer


# JPEG encoder
def JPEGencode (img, subimg, qScale, zeros):

    # Use JPEG library functions to encode each block
    def encode_block(block, qTable, qScale, DCpred, huffTables, qBlocks, runLengths):

        dctBlock = j.blockDCT(block)                        # DCT
        qBlock = j.quantizeJPEG(dctBlock, qTable, qScale)   # Quantization
        qBlocks.append(qBlock.flatten())
        runSymbols = j.runLength(qBlock, DCpred)            # RLE
        runLengths.append(runSymbols)
        DCpred = qBlock[0, 0]                               # Set DC prediction for next block
        huffStream = j.huffEnc(runSymbols, huffTables)      # Huffman encoding

        return huffStream, DCpred
    
    # Function to calculate Q entropy
    def q_entropy(qblocks):

        q_values = np.concatenate([qblock for qblock in qblocks])
        # Counting unique qblock values
        _, counts = np.unique(q_values, return_counts=True)
        probabilities = counts / len(q_values)

        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    # Function to calculate RLE entropy
    def rle_entropy(runLengths):
        
        rle_values = []

        # Extracting the DC values and the pairs
        for rle in runLengths:
            pairs = rle[1:]
            for pair in pairs:
                rle_values.append(pair)

        # Counting unique dc values and pairs
        _, counts = np.unique(rle_values, axis=0, return_counts=True)
        probabilities = counts / len(rle_values)

        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    # Function to include zeros in the Quantization tables
    def input_zeros(array, zeros):

        if zeros == 0:
            return array
        
        sequence = utils.zig_zag(array)
        sequence[-zeros:] = 999
        sequence = np.insert(sequence, 0, array[0,0])
        array = utils.izig_zag(sequence, (8, 8))

        return array        

    # Convert rgb image to Y, Cr, Cb channels
    imageY, imageCr, imageCb = j.convert2ycrcb(img, subimg)

    # Images to 8x8 blocks
    blocksY = utils.to_blocks(imageY)
    blocksCr = utils.to_blocks(imageCr)
    blocksCb = utils.to_blocks(imageCb)

    # Quantization tables
    qTableL = input_zeros(utils.Q_y, zeros)
    qTableC = input_zeros(utils.Q_c, zeros)

    # Huffman tables
    DCL = utils.dc_luminance
    DCC = utils.dc_chrominance
    ACL = utils.ac_luminance
    ACC = utils.ac_chrominance

    # Init list for output
    JPEGenc = []
    qBlocks = []    # to calculate Q entropy
    runLengths = [] # to calculate RLE entropy
    totalBits = 0

    # Encoding Luminance blocks
    DCpredY = 0
    _, W = imageY.shape
    
    for i in range(len(blocksY)):

        # Encode Y block
        huffStream, DCpredY = encode_block(blocksY[i], qTableL, qScale, DCpredY, (DCL, ACL), qBlocks, runLengths)
        totalBits += len(huffStream)
        # Get coordinates of Y block
        indHor = i % (W / 8) + 1
        indVer = i // (W / 8) + 1
        # Write Y block information
        encBlockY = block('Y', indHor, indVer, huffStream)
        JPEGenc.append(encBlockY)

    # Encoding Chrominance blocks
    DCpredCr = 0
    DCpredCb = 0
    _, W = imageCr.shape

    for i in range(len(blocksCr)):

        # Encode Cr block
        huffStream, DCpredCr = encode_block(blocksCr[i], qTableC, qScale, DCpredCr, (DCC, ACC), qBlocks, runLengths)
        totalBits += len(huffStream)
        # Get coordinates of Cr block
        indHor = i % (W / 8) + 1 
        indVer = i // (W / 8) + 1
        # Write Cr block information
        encBlockCr = block('Cr', indHor, indVer, huffStream)
        JPEGenc.append(encBlockCr)

        # Encode Cb block
        huffStream, DCpredCb = encode_block(blocksCb[i], qTableC, qScale, DCpredCb, (DCC, ACC), qBlocks, runLengths)
        totalBits += len(huffStream)
        # Get coordinates of Cb block
        indHor = i % (W / 8) + 1
        indVer = i // (W / 8) + 1
        # Write Cb block information
        encBlockCb = block('Cb', indHor, indVer, huffStream)
        JPEGenc.append(encBlockCb)


    print('-Total encoded bits:', totalBits)
    print('-Q entropy =', round(q_entropy(qBlocks), 4))
    print('-RLE entropy =', round(rle_entropy(runLengths), 4))
    
    # Metadata
    data = metadata(qScale, qTableL, qTableC, DCL, DCC, ACL, ACC, totalBits) # output
    JPEGenc.insert(0, data) # first component of output tuple

    return tuple(JPEGenc)


# JPEG decoder
def JPEGdecode(JPEGenc):
    
    meta = JPEGenc[0]

    DCpredY = 0
    DCpredCr = 0
    DCpredCb = 0
    maxHor = np.zeros(3)
    maxVer = np.zeros(3)
    blocksY = []
    blocksCr = []
    blocksCb = []

    # Decoding process
    for blk in JPEGenc[1:]:

        # Decoding Y blocks
        if blk.blkType == 'Y':
            blockY, DCpredY, indHor, indVer = blk.decode((meta.DCL, meta.ACL), DCpredY, meta.qTableL, meta.qScale)
            maxHor[0] = max(maxHor[0], indHor)
            maxVer[0] = max(maxVer[0], indVer)
            blocksY.append((blockY, indHor, indVer))

        # Decoding Cr blocks
        elif blk.blkType == 'Cr':
            blockCr, DCpredCr, indHor, indVer = blk.decode((meta.DCC, meta.ACC), DCpredCr, meta.qTableC, meta.qScale)
            maxHor[1] = max(maxHor[1], indHor)
            maxVer[1] = max(maxVer[1], indVer)
            blocksCr.append((blockCr, indHor, indVer))
            
        # Decoding Cb blocks
        elif blk.blkType == 'Cb':
            blockCb, DCpredCb, indHor, indVer = blk.decode((meta.DCC, meta.ACC), DCpredCb, meta.qTableC, meta.qScale)
            maxHor[2] = max(maxHor[2], indHor)
            maxVer[2] = max(maxVer[2], indVer)
            blocksCb.append((blockCb, indHor, indVer))

    # Sort decoded blocks based on coordinates
    sorted_blocksY = sorted(blocksY, key=lambda k: (k[2], k[1]))
    sorted_blocksCr = sorted(blocksCr, key=lambda k: (k[2], k[1]))
    sorted_blocksCb = sorted(blocksCb, key=lambda k: (k[2], k[1]))

    # Remove coordinate information from sorted blocks
    blocksY_data = [block[0] for block in sorted_blocksY]
    blocksCr_data = [block[0] for block in sorted_blocksCr]
    blocksCb_data = [block[0] for block in sorted_blocksCb]

    # Calculate shapes from max coordinates
    shapeY = (maxVer[0] * 8, maxHor[0] * 8)
    shapeCr = (maxVer[1] * 8, maxHor[1] * 8)
    shapeCb = (maxVer[2] * 8, maxHor[2] * 8)

    # Blocks back to image shapes.0
    imageY = utils.from_blocks(blocksY_data, shapeY)
    imageCr = utils.from_blocks(blocksCr_data, shapeCr)
    imageCb = utils.from_blocks(blocksCb_data, shapeCb)

    # Define subsampling type
    if (shapeCr[1] < shapeY[1]):
        subimg = [4, 2, 2]
        if (shapeCr[0] < shapeY[0]):
            subimg = [4, 2, 0]
    else:
        subimg = [4, 4, 4]

    # Convert back to RGB
    imgRec = j.convert2rgb(imageY, imageCr, imageCb, subimg)
   
    return imgRec
