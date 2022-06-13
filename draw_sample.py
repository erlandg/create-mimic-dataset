import sys
import glob

import numpy as np
import matplotlib.pyplot as plt

import pydicom as dicom
import cv2
from imutils import rotate
from skimage.transform import resize
import utils.dicom_processing as preprocess


fnames = glob.glob("physionet.org/**/*.dcm", recursive=True)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
for i in range(int(sys.argv[1])):
    idx = np.random.choice(range(len(fnames)))
    fdicom = dicom.dcmread(fnames[idx], force=True).pixel_array
    rotated = preprocess.auto_rotate(fdicom)
    plt.imsave(
        f"samples/{i}_image.png",
        preprocess._normalise_img(resize(fdicom, (512,512))),
        vmin=0,
        vmax=1,
        cmap="binary_r",
    )
    plt.imsave(
        f"samples/{i}_image_rot.png",
        preprocess._normalise_img(resize(rotated, (512,512))),
        vmin=0,
        vmax=1,
        cmap="binary_r",
    )
    plt.imsave(
        f"samples/{i}_image_proc.png",
        preprocess.preprocess(fdicom, clahe, size=(512,512), bits=8),
        vmin=0,
        vmax=1,
        cmap="binary_r",
    )
