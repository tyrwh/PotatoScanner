import marimo

from ultralytics import YOLO
from ultralytics.engine.results import Masks
import skimage
import re
import pandas as pd
import numpy as np
import cv2
from skimage.restoration import inpaint
from skimage.filters import threshold_mean
from scipy.spatial import ConvexHull
from pathlib import Path

model = YOLO('./yolo11x_seg_best.pt')
model.to(device='cuda')

def inpaint_to_height(img):
    mask = img == 0
    # skimage will mess up uint16 input for some reason
    # convert to float64 then back to uint16
    depth_mm = inpaint.inpaint_biharmonic(img.astype(np.float64), mask).astype(np.uint16)
    height_mm = np.max(depth_mm) - depth_mm
    return height_mm

class SingleTuberImgPair():
    def __init__(self, res):
        self.res = res
        self.rgb_path = Path(res.path)
        # add the depth path, based on the rgb path
        parts = list(self.rgb_path.parts)
        if len(parts) > 2 and parts[-2] == "rgb":
            parts[-2] = "depth"
        self.depth_path = Path(*parts)
        self.rgb_png = skimage.io.imread(str(self.rgb_path))
        self.depth_png = skimage.io.imread(str(self.depth_path))
    def calc_cls_proportions(self):
        np_mask = self.res.masks.data.numpy()
        # empty dict to fill in
        cls_proportions = {cls:np.float32(0.0) for cls in self.res.names.values()}
        px_sum = np.sum(np_mask)
        # update with sums from the identified classes
        # note that multiple masks can have same class, use +=
        for i,cls in enumerate(self.res.boxes.cls):
            cls_proportions[self.res.names[int(cls)]] += np.sum(np_mask[i,:,:])/px_sum
        self.cls_proportions = cls_proportions
    def mask_fg(self):
        # pull keys of nonsprout classes
        nonsprout_keys = [k for k in self.res.names.keys() if self.res.names[k] != 'Sprout']
        # then get the indices of masks that correspond to these keys
        nonsprout_idx = [i for i,x in enumerate(self.res.boxes.cls.numpy()) if x in nonsprout_keys]
        nonsprout_masks = self.res.masks.data[nonsprout_idx,:,:]
        fg_mask = np.sum(nonsprout_masks.numpy(),0) > 0
        self.fg_mask = skimage.transform.resize(fg_mask, self.res.orig_shape)
        
    def calc_tuber_size(self, mm_per_pixel = 304/1750):
        # very rough guess on the mm per pixel, assuming that the conveyor belt is about a foot wide
        # 304 mm in a foot, 1750-ish pixels across on the belt
        tuber_metrics = {}
        height_mm = inpaint_to_height(self.depth_png)
        bg_zeroed = np.where(self.fg_mask, height_mm, 0)
        # actual volume calc
        volume_cm3 = np.sum(bg_zeroed) * (mm_per_pixel ** 2) / 1000
        tuber_metrics['volume_cm3'] = volume_cm3
        # reformat the data from [[z,z],[z,z]] to [[x,y,z],[x,y,z]]
        nonzero_inds = np.argwhere(bg_zeroed > 0).astype(np.uint16)
        nonzero_heights = bg_zeroed[bg_zeroed > 0].astype(np.uint16)
        # you need to have points for both the top surface profile and the base
        # otherwise the hull volume will be underestimated
        top_3d_coords = np.column_stack((nonzero_inds,
                                         nonzero_heights))
        base_3d_coords = np.column_stack((nonzero_inds,
                                         [0] * nonzero_inds.shape[0]))
        points_3d = np.row_stack((top_3d_coords, base_3d_coords))
        # now can fit a convex hull and pull volume from that
        tuber_hull_3d = ConvexHull(points_3d)
        hull_volume_cm3 =  tuber_hull_3d.volume * (mm_per_pixel ** 2) / 1000
        tuber_metrics['hull_volume_cm3'] = hull_volume_cm3
        tuber_metrics['solidity'] = volume_cm3 / hull_volume_cm3
        self.tuber_metrics = tuber_metrics


class TuberImgSet():
    def __init__(self, base_img_path):
        self.base_img_path = Path(base_img_path)
        self.rgb_img_paths = sorted(self.base_img_path.glob('Tracked_Images/rgb/*.png'))
        self.depth_img_paths = sorted(self.base_img_path.glob('Tracked_Images/depth/*.png'))
    def run_yolo_seg(self, model, verbose=False):
        print('Running YOLO11x-seg on dir: %s' % self.base_img_path)
        self.yolo_res = model.predict(self.base_img_path / 'Tracked_Images/rgb', verbose=verbose)
    def calc_all_metrics(self):
        print('Calculating metrics...')
        all_cls_props = []
        all_sizes = []
        for res in self.yolo_res:
            img_pair = SingleTuberImgPair(res)
            img_pair.calc_cls_proportions()
            img_pair.mask_fg()
            img_pair.calc_tuber_size()
            all_cls_props.append(img_pair.cls_proportions)
            all_sizes.append(img_pair.tuber_metrics)
        props = pd.DataFrame(all_cls_props)
        sizes = pd.DataFrame(all_sizes)
        df = props.join(sizes)
        df.insert(0,'dir',self.base_img_path)
        df.insert(1,'img', [x.name for x in self.rgb_img_paths])
        self.df = df

foo = TuberImgSet('sample_images/O23-0765/')
foo.run_yolo_seg(model)
foo.calc_all_metrics()


