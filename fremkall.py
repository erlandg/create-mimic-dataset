import pydicom as dicom
import matplotlib.pyplot as plt
import glob

fnames = glob.glob("*.dcm")
dss = [dicom.dcmread(fname) for fname in fnames]
for ds, fname in zip(dss, fnames):
	plt.imsave(f'{fname}.png', ds.pixel_array, cmap='binary_r')
