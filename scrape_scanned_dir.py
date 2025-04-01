from ultralytics import YOLO
from ultralytics.engine.results import Masks
import skimage
import pandas as pd
import numpy as np
from skimage.restoration import inpaint
from skimage.filters import threshold_mean
from scipy.spatial import ConvexHull
from pathlib import Path
from torch import cuda

import argparse

parser = argparse.ArgumentParser(description="Process tuber images and calculate metrics.")
parser.add_argument('-i', '--input', required=True, help="Input directory containing images.")
parser.add_argument('-o', '--output', required=True, help="Output CSV file name (must end with .csv).")
parser.add_argument('-m', '--model', required=True, help="YOLO model file (.pt format).")

args = parser.parse_args()

if not args.output.endswith('.csv'):
    raise ValueError("Output file name must end with .csv")
if not args.model.endswith('.pt'):
    raise ValueError("Model file must be in .pt format")

def validate_target_dir(target_dir):
    if not isinstance(target_dir, Path):
        raise TypeError("target_dir must be a Path object")
    if not target_dir.exists():
        raise FileNotFoundError(f"The directory {target_dir} does not exist")
    if not target_dir.is_dir():
        raise NotADirectoryError(f"The path {target_dir} is not a directory")
    tracked_images_dir = target_dir / "Tracked_Images"
    if tracked_images_dir.exists() and tracked_images_dir.is_dir():
        return [target_dir]
    else:
        subdirs_with_tracked_images = []
        for subdir in target_dir.iterdir():
            if subdir.is_dir() and (subdir / "Tracked_Images").exists() and (subdir / "Tracked_Images").is_dir():
                subdirs_with_tracked_images.append(subdir)
        return subdirs_with_tracked_images if subdirs_with_tracked_images else None


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
    def append_notes(self):
        col_headers = ["Type", "Knobs", "Tuber bend", "Sprouting", "Growth cracks", "Eye depth", "Eye number", "Russeting", "Aligator hide", "Skin defect", "Greening"]
        for header in col_headers:
            self.df[header] = ""  # Initialize columns with empty strings
        notes_path = self.base_img_path / 'notes.txt'
        if not notes_path.exists():
            raise Warning(f"Notes file {notes_path} does not exist.")
        else:
            with open(notes_path, 'r') as f:
                notes = f.readlines()
                for line in notes:
                    for header in col_headers:
                        if line.startswith(header):
                            self.df[header] = line.split(":")[-1].strip()
                            break

def main():
    model = YOLO(args.model)
    if cuda.is_available():
        model.to(device='cuda')
    target_subdirs = validate_target_dir(Path(args.input))
    if target_subdirs is None:
        raise ValueError("No subdirectories with tracked images found.")
    all_dfs = []
    for target_subdir in target_subdirs:
        img_set = TuberImgSet(target_subdir)
        if len(img_set.rgb_img_paths) != len(img_set.depth_img_paths):
            print(f"Skipping {target_subdir} due to mismatched RGB and depth image counts.")
            continue
        if len(img_set.rgb_img_paths) == 0 or len(img_set.depth_img_paths) == 0:
            print(f"Skipping {target_subdir} due to missing RGB or depth images.")
            continue
        img_set.run_yolo_seg(model)
        img_set.calc_all_metrics()
        img_set.append_notes()
        all_dfs.append(img_set.df)
    all_dfs = pd.concat(all_dfs, ignore_index=True)
    all_dfs.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")
    # print the first 5 rows of the dataframe
    print(all_dfs.head())

if __name__ == "__main__":
    main()
