#Transformation for conversion between patient coordinates in millimeters and array coordinates
from collections import namedtuple
import numpy as np

IrcTuple = collections.namedtuple('IrcTuple', ['index', 'row', 'col'])
XyzTuple = collections.namedtuple('XyzTuple', ['x', 'y', 'z'])

def irc2xyz(coord_irc, origin_xyz, vxSize_xyz, direction_a):
	cri_a = np.array(coord_irc)[::-1] #swaps the order while we convert to a Numpy array
	origin_a = np.array(origin_xyz)
	vxSize_a = np.array(vxSize_xyz)
	coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a
	return XyzTuple(coords_xyz)

def xyz2irc(coord_xyz, origin_xyz, vxSize_xyz, direction_a):
	origin_a = np.array(origin_xyz)
	vxSize_a = np.array(vxSize_xyz)
	coord_a = np.array(coord_xyz)
	cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vxSize_a
	cri_a = np.round(cri_a)
	return IrcTuple(int(cri_a[2]), int(cri_a[1]), int(cri_a[0]))

def importstr(module_str, from_=None):
    """
    >>> importstr('os')
    <module 'os' from '.../os.pyc'>
    >>> importstr('math', 'fabs')
    <built-in function fabs>
    """
    if from_ is None and ':' in module_str:
        module_str, from_ = module_str.rsplit(':')

    module = __import__(module_str)
    for sub_str in module_str.split('.')[1:]:
        module = getattr(module, sub_str)

    if from_:
        try:
            return getattr(module, from_)
        except:
            raise ImportError('{}.{}'.format(module_str, from_))
    return module

