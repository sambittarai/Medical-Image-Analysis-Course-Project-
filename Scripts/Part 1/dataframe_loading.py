import numpy as np
import pandas as pd
from collections import namedtuple


CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'isNodule_bool, diameter_mm, series_uid, center_xyz',
)

@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    # We construct a set with all series_uids that are present on disk.
    # This will let us use the data, even if we haven't downloaded all of
    # the subsets yet.
    mhd_list = glob.glob('/content/drive/MyDrive/LUNA 16/Dataset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    diameter_dict = {}
    with open('/content/drive/MyDrive/LUNA 16/Dataframes/annotations.csv', "r") as f:
      for row in list(csv.reader(f))[1:]:
        series_uid = row[0]
        annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
        annotationDiameter_mm = float(row[4])

        diameter_dict.setdefault(series_uid, []).append(
          (annotationCenter_xyz, annotationDiameter_mm))
        
    candidateInfo_list = []
    with open('/content/drive/MyDrive/LUNA 16/Dataframes/candidates.csv', "r") as f:
      for row in list(csv.reader(f))[1:]:
        series_uid = row[0]

        #If a series_uid is not present, it's in a subset we don't have on disk, so we should skip it.
        if series_uid not in presentOnDisk_set and requireOnDisk_bool:
          continue

        isNodule_bool = bool(int(row[4]))
        candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

        candidateDiameter_mm = 0.0

        #For each of the candidate entries for a given series_uid, we loop through the annotations we collected earlier for the same series_uid and see if the two coordinates are close enough to consider them the nodule. 
        #If we don't find a match than we will treat the nodule as having 0 diameter.
        for annotation_tup in diameter_dict.get(series_uid, []):
          annotationCenter_xyz, annotationDiameter_mm = annotation_tup
          for i in range(3):
            delta_mm = abs(candidateCenter_xyz[i] - annotationCenter_xyz[i])
            # Divides the diameter by 2 to get the radius, and divides the radius by 2 to require that the two nodule center points not be too far apart relative to the size of the nodule.
            if delta_mm > annotationDiameter_mm / 4: 
              break

          else:
            candidateDiameter_mm = annotationDiameter_mm
            break

        candidateInfo_list.append(CandidateInfoTuple(
          isNodule_bool,
          candidateDiameter_mm,
          series_uid,
          candidateCenter_xyz))

    # This means we have all of the actual nodule samples starting with the largest first, followed by all of the non-nodule samples (which don't have nodule size information).
    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list

'''
#Testing
Cadidates = getCandidateInfoList()
'''
