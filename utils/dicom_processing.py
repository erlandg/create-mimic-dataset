import numpy as np
from imutils import rotate
from skimage.transform import resize
from skimage.exposure import adjust_sigmoid
import pydicom as dicom



def _normalise_img(img):
    return (img - img.min())/(img.max() - img.min())


def preprocess(input, CLAHE, size=(256,256), bits=16):
    if bits == 16:
        dtype = np.uint16
    elif bits == 8:
        dtype = np.uint8
    pixel_array = auto_rotate(input)
    pixel_array = _normalise_img(pixel_array)
    pixel_array = CLAHE.apply((pixel_array*(2**bits-1)).astype(dtype))/(2**bits-1)
    pixel_array = resize(pixel_array, size)
    return _normalise_img(pixel_array)


def preprocess_dicom(path, clahe, **kwargs):
    try:
        dicom_ = dicom.dcmread(path, force=True).pixel_array
    except:# (AttributeError, FileNotFoundError):
        return None
    return preprocess(dicom_, clahe, **kwargs)



def auto_rotate(image, min_empty_rows=20, max_tilt=45-11.25, step=45/16):
    best_im = image
    best_N = 0

    # Checks how many of the first non-zero rows and columns are non-zero.
    # Gives an indication of tilt. By only accepting the ones above a threshold,
    # we avoid problems such as black squares missing in corner.
    first_five_0 = np.where(np.any(image, axis=1))[0][:int(10)]
    last_five_0 = np.where(np.any(image, axis=1))[0][-int(10):]
    mean0 = (np.any(image[first_five_0], axis=0).mean() + \
        np.any(image[last_five_0], axis=0).mean())/2
    first_five_1 = np.where(np.any(image, axis=0))[0][:int(10)]
    last_five_1 = np.where(np.any(image, axis=0))[0][-int(10):]
    mean1 = (np.any(image[:,first_five_1], axis=1).mean() + \
        np.any(image[:,last_five_1], axis=1).mean())/2

    if mean0 > .25 and mean1 > .25:
        best_im = image
        best_N = (image.sum(0) == 0).sum() + (image.sum(1) == 0).sum()
    else:
        for rotation in np.arange(-max_tilt+step, max_tilt-(step-1e-8), step):
            scale = max(np.sin(np.radians(rotation)), np.cos(np.radians(rotation)))
            if rotation != 0:
                rot = rotate(image, angle=rotation, scale=scale)
            else:
                rot = image
            empty_cols = (rot.sum(0) == 0).sum()
            empty_rows = (rot.sum(1) == 0).sum()
            if empty_cols + empty_rows < min_empty_rows:
                continue
            if empty_rows + empty_cols > best_N:
                best_im = rot
                best_N = empty_rows + empty_cols

    if best_N >= min_empty_rows:
        ax0 = np.any(best_im, axis=1)
        first0, last0 = np.where(ax0)[0][[0,-1]]
        ax1 = np.any(best_im, axis=0)
        first1, last1 = np.where(ax1)[0][[0,-1]]
        return best_im[first0:last0+1, first1:last1+1]
    return image
