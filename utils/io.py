import numpy as np
import cv2
import re
from struct import pack, unpack
import sys        

def read_pfm(pfm_file_path: str)-> np.ndarray:
    color = None
    width = None
    height = None
    scale = None
    data_type = None
    with open(pfm_file_path, 'rb') as file:
        header = file.readline().decode('UTF-8').rstrip()

        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('UTF-8'))

        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        # scale = float(file.readline().rstrip())
        scale = float((file.readline()).decode('UTF-8').rstrip())
        if scale < 0: # little-endian
            data_type = '<f'
        else:
            data_type = '>f' # big-endian
        data_string = file.read()
        data = np.fromstring(data_string, data_type)
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = cv2.flip(data, 0)
    return data

def write_pfm(fp, image, scale=1):
    color = None
    image = np.flipud(image)
    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    with open(fp, 'wb') as f:
        header = 'PF\n' if color else 'Pf\n'
        shape = '%d %d\n' % (image.shape[1], image.shape[0])
        fp.write(header.encode('utf-8'))
        fp.write(shape.encode('utf-8'))

        endian = image.dtype.byteorder

        if endian == '<' or endian == '=' and sys.byteorder == 'little':
            scale = -scale

        scale = '%f\n' % scale
        f.write(scale.encode('utf-8'))

        image_string = image.tostring()
        f.write(image_string.encode('utf-8'))

def read_dmb(fp):
    '''read Gipuma .dmb format image'''

    with open(fp, "rb") as fid:
        
        image_type = unpack('<i', fid.read(4))[0]
        height = unpack('<i', fid.read(4))[0]
        width = unpack('<i', fid.read(4))[0]
        channel = unpack('<i', fid.read(4))[0]
        
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channel), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

def write_dmb(fp, image):
    '''write Gipuma .dmb format image'''
    
    image_shape = np.shape(image)
    width = image_shape[1]
    height = image_shape[0]
    if len(image_shape) == 3:
        channels = image_shape[2]
    else:
        channels = 1

    if len(image_shape) == 3:
        image = np.transpose(image, (2, 0, 1)).squeeze()

    with open(fp, "wb") as fid:
        # fid.write(pack(1))
        fid.write(pack('<i', 1))
        fid.write(pack('<i', height))
        fid.write(pack('<i', width))
        fid.write(pack('<i', channels))
        image.tofile(fid)
    return 

def read_projection(fp):
    pass

def write_projection(fp: str, projection_matrix: np.array):
    with open(fp, "w") as f:
        for i in range(0, 3):
            for j in range(0, 4):
                f.write(str(projection_matrix[i][j]) + ' ')
            f.write('\n')
        f.write('\n')
        f.close()


def read_camera(camera_file_path: str)-> tuple:
    '''
    Loads camera from path
        Camera format:
            extrinsic
            E11 E12 E13 E14
            E21 E22 E23 E24
            E31 E32 E33 E34
            E41 E42 E43 E44

            intrinsic
            K11 K12 K13
            K21 K22 K23
            K31 K32 K33

            T1 T2 ...

    Arguments:
        camera_file_path: path to camera file
    Return:
        Extrinsic: 4x4 numpy array containing R | t
        Intrinsic: 3x3 numpy array containing intrinic matrix
        Range: min, interval, num_interval, max
    '''
    with open(camera_file_path, 'r') as f:
        lines = f.readlines()

        E = np.array([
            [float(v) for v in line.split()] for line in lines[1:5]
        ], dtype=np.float32)

        K = np.array([
            [float(v) for v in line.split()] for line in lines[7:10]
        ], dtype=np.float32)

        tokens = [float(v) for v in lines[11].split()]
    return K, E, tokens

def write_camera(fp:str, 
                 intrinsic: np.ndarray, 
                 extrinsic: np.ndarray, 
                 tokens:np.ndarray):
    '''
    Writes camera to path given intrinsics / extrinsics / tokens
        Camera format:
            extrinsic
            E11 E12 E13 E14
            E21 E22 E23 E24
            E31 E32 E33 E34
            E41 E42 E43 E44

            intrinsic
            K11 K12 K13
            K21 K22 K23
            K31 K32 K33

            T1 T2 ...
    Arguments:
        fp(str): path to camera file
        intrinsic(np.ndarray): 3x3 camera intrinsic matrix
        extrinsic(np.ndarray): 3x3 camera intrinsic matrix
        tokens(np.ndarray): Nx1 tokens for extra data
    '''
    k_str= '\n'.join([' '.join([str(v) for v in l]) for l in intrinsic])
    e_str = '\n'.join([' '.join([str(v) for v in l]) for l in extrinsic])
    t_str = ' '.join([str(t) for t in tokens])
    with open(fp, 'w') as f:
        f.write('extrinsic\n')
        f.write(e_str)
        f.write('\n')
        f.write('\n')
        f.write('intrinsic\n')
        f.write(k_str)
        f.write('\n')
        f.write('\n')
        f.write(t_str)
