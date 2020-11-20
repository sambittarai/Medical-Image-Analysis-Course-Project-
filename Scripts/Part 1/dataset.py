import numpy as np
import pandas as pd
import glob
import os
from collections import namedtuple
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from util.utils import XyzTuple, xyz2irc
from util.logconf import logging


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'isNodule_bool, diameter_mm, series_uid, center_xyz',
)

@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    # We construct a set with all series_uids that are present on disk.
    # This will let us use the data, even if we haven't downloaded all of
    # the subsets yet.
    mhd_list = glob.glob('data-unversioned/part2/luna/subset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    diameter_dict = {}
    with open('data/part2/luna/annotations.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])

            diameter_dict.setdefault(series_uid, []).append(
                (annotationCenter_xyz, annotationDiameter_mm)
            )

    candidateInfo_list = []
    with open('data/part2/luna/candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue

            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

            candidateDiameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotationCenter_xyz, annotationDiameter_mm = annotation_tup
                for i in range(3):
                    delta_mm = abs(candidateCenter_xyz[i] - annotationCenter_xyz[i])
                    if delta_mm > annotationDiameter_mm / 4:
                        break
                else:
                    candidateDiameter_mm = annotationDiameter_mm
                    break

            candidateInfo_list.append(CandidateInfoTuple(
                isNodule_bool,
                candidateDiameter_mm,
                series_uid,
                candidateCenter_xyz,
            ))

    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list

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

		self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
		self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
		self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)
		#The above are the inputs we need to pass into the xyz2irc conversion function

	def getRawCandidate(self, center_xyz, width_irc):
		'''
		The getRawNodule function takes the center expressed in the patient coordinate system (X,Y,Z) as well as width in voxels,
		And returns a cubic chunk of CT as well as the center of the candidate converted to array coordinates.
		'''
		center_irc = xyz2irc(
			center_xyz,
			self.origin_xyz,
			self.vxSize-xyz,
			self.direction_a)

		slice_list = []
		for axis, center_val in enumerate(center_irc):
			start_ndx = int(round(center_val - width_irc[axis]/2))
			end_ndx = int(start_ndx + width_irc[axis])

			assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

			if start_ndx < 0:
				#Crop outside of CT array
				start_ndx = 0
				end_ndx = int(width_irc[axis])

			if end_ndx > self.hu_array.shape[axis]:
				#Crop outside of CT array
				end_ndx = self.hu_array.shape[axis]
				start_ndx = int(self.hu_array.shape[axis] - width_irc[axis])

			slice_list.append(slice(start_ndx, end_ndx))

		ct_chunk = self.hu_array[tuple(slice_list)]

		return ct_chunk, center_irc

@functools.lru_cache(1, typed=True)
def getCt(series_uid):
	return Ct(series_uid)

@raw_cache.memorize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
	ct = getCt(series_uid)
	ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
	return ct_chunk, center_irc

class LunaDataset(Dataset):

	def __init__(self, val_stride=0, isValSet_bool=None, series_uid=None):
		self.candidateInfo_list = copy.copy(getCandidateInfoList()) #Copies the return value so the cached copy won't be impacted by altering self.candidateInfo_list

		if series_uid:
			self.candidateInfo_list = [
			x for x in self,candidateInfo_list if x.series_uid == series_uid]

		if isValSet_bool:
			assert val_stride > 0, val_stride
			self.candidateInfo_list = self.candidateInfo_list[::val_stride]
			assert self.candidateInfo_list

		elif val_stride > 0:
			del self.candidateInfo_list[::val_stride]
			assert self.candidateInfo_list

		log.info("{!r}: {} {} samples".format(self, len(self.candidateInfo_list), "validation" if isValSet_bool else "training"))

		def __len__(self):
			return len(self.candidateInfo_list)

		def __getitem__(self, ndx):
			candidateInfo_tup = self.candidateInfo_list[ndx]
			width_irc = (32, 48, 48)

			candidate_a, center_irc = getCtRawCandidate(
				candidateInfo_tup.series_uid,
				candidateInfo_tup.center_xyz,
				width_irc)

			candidate_t = torch.from_numpy(candidate_a)
			candidate_t = candidate_t.to(torch.float32)
			candidate_t = candidate_t.unsqueeze(0)

			pos_t = torch.tensor([
				not candidateInfo_tup.isNodule_bool,
				candidateInfo_tup.isNodule_bool],
				dtype=torch.long)

			return (candidate_t, pos_t, candidateInfo_tup.series_uid, torch.tensor(center_irc))



	



