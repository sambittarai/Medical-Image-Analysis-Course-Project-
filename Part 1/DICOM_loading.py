import numpy as np
import glob
import SimpleITK as sitk

class CT:

def __init__(self, series_uid):
	mhd_path = glob.glob('/content/drive/MyDrive/LUNA 16/Dataset/{}.mhd'.format(series_uid))[0]
	#sitk.ReadImage implicitly consumes the .raw file in addition to the passed in .mhd file.
	ct_mhd = sitk.ReadImage(mhd_path)
	#Creates a numpy array
	ct_array = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
	#pixel clipping
	ct_array.clip(-1000, 1000, ct_array)

	self.series_uid = series_uid
	self.hu_array = ct_array
	self.spacing = ct_mhd.GetSpacing()