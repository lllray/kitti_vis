import numpy as np
import sys
import os
from libtiff import TIFF
from PIL import Image
from calibration_kitti import Calibration
import struct
# to open a tiff file for reading:
baseline = 0.1
step = 2
def disp_to_rgbcloud(disp_image,color_image,cloud_file,calib_file):
    disp_array = np.array(disp_image)
    color_array = np.array(color_image)
    shape = disp_array.shape
    calib = Calibration(calib_file)
    focus = calib.fu
    cx = calib.cu
    cy = calib.cv
    fb = baseline * focus
    rgbcloud = np.empty(shape=[int(shape[0] * shape[1]/step/step),5],dtype=np.float32)
    for i in range(0, shape[0], step):
        for j in range(0, shape[1], step):
            disp = disp_array[i, j]
            value = color_array[i, j]
            z = fb / disp
            y = (i - cy) * z / focus
            x = (j - cx) * z / focus
            value[3] = 0
            rgb = value.view(np.float32)[0]
            # RGB = rgb.view((np.uint8, 4)).astype(np.float32)
            id = int(i/step)*int(shape[1]/step) + int(j/step)
            rgbcloud[id] = np.array([x,y,z,z,rgb])
    #print(rgbcloud[0])
    #print(rgbcloud[256000-1])
    rgbcloud[:, 0:3] = calib.bev_img_to_ground_lidar(rgbcloud[:, 0:3])
    print(rgbcloud[0])
    #print(rgbcloud[0])
    #print(rgbcloud[256000-1])
    cloud_file.write(rgbcloud)
print('Usage: python3 convert_disp_to_depth.py disp_dir')
disp_dir = sys.argv[1]
color_dir = disp_dir + '/../image_2'
cloud_dir = disp_dir + '/../stereo_cloud'
calib_dir = disp_dir + '/../calib'
if not os.path.exists(cloud_dir):
    os.makedirs(cloud_dir)
file  = os.listdir(disp_dir)

for image in file:
    suffix = image.strip().split('.')[-1]
    timestamp = image.strip().split('.')[0]
    if suffix == 'tiff':
        print('Process image',image)
        tif = TIFF.open(disp_dir+'/'+image, mode='r')
        disp_image = tif.read_image()
        color = Image.open(color_dir + '/' + timestamp + ".png")
        cloud_file = open(cloud_dir+ '/' + timestamp + ".bin",'ab+')
        calib_file = calib_dir+ '/' + timestamp + ".txt"
        disp_to_rgbcloud(disp_image,color,cloud_file,calib_file)

