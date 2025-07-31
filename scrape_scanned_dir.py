from ultralytics import YOLO
from ultralytics.engine.results import Masks
import skimage
import pandas as pd
import numpy as np
import cv2
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
parser.add_argument('-a', '--annotate-dir', required=False, help="Optional directory to save annotated images with bounding boxes.")

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

# from a cv2-generated contour, rotate it to be oriented vertically (long axis pointed up and down)
# note: produces a rotated MASK, not a contour
def rotate_contour(cnt):
    (center_x,center_y),(MA,ma),angle = cv2.fitEllipse(cnt)
    xmin = np.min(cnt[:,:,0])
    ymin = np.min(cnt[:,:,1])
    w = np.max(cnt[:,:,0]) - xmin
    h = np.max(cnt[:,:,1]) - ymin
    drawn_cnt = np.zeros((h+w,h+w), dtype = np.uint8)
    cv2.drawContours(drawn_cnt, [cnt - [xmin,ymin] + [int(h/2),int(w/2)]], 0, 255, cv2.FILLED)
    rot_matrix = cv2.getRotationMatrix2D((int((w+h)/2), int((w+h)/2)), angle, 1)
    rotated = cv2.warpAffine(drawn_cnt, rot_matrix, (h+w,h+w))
    return rotated

def oriented_bounding_box(cnt):
    # from a cv2-generated contour, rotate and create an oriented bounding box in same format as tuple created by cv2.minAreaRect()
    # could expedite by integrating rotate_contour() above, but nice that this is standalone
    (center_x,center_y),(MA,ma),angle = cv2.fitEllipse(cnt)
    xmin = np.min(cnt[:,:,0])
    ymin = np.min(cnt[:,:,1])
    w = np.max(cnt[:,:,0]) - xmin
    h = np.max(cnt[:,:,1]) - ymin
    # make a canvas large enough to fit any arbitrary contour/rotated contour, redraw the contour on it, then rotate
    # draw it such that the center of the ellipse-of-best-fit is exactly at the midpoint
    drawn_cnt = np.zeros((h+w,h+w), dtype = np.uint8)
    cv2.drawContours(drawn_cnt, [cnt - [int(center_x),int(center_y)] + [int((w+h)/2),int((h+w)/2)]], 0, 255, cv2.FILLED)
    rot_matrix = cv2.getRotationMatrix2D(((h+w)/2, (h+w)/2), angle, 1)
    rotated_aliased = cv2.warpAffine(drawn_cnt, rot_matrix, (h+w,h+w))
    # rotating applies anti-aliasing - filter to avoid this
    _, rotated = cv2.threshold(rotated_aliased, 127, 255, cv2.THRESH_BINARY)
    # redraw contours in order to find oriented bounding box (OBB)
    cnt_rotated, hierarchy = cv2.findContours(rotated, 0, 2)
    obb_x,obb_y,obb_w,obb_h = cv2.boundingRect(cnt_rotated[0])
    # offset vector between oriented rectangle center and ellipse center
    # tends to be small, but makes the annotations cleaner
    center_offset = np.array((obb_x + (obb_w/2) - (w+h)/2, obb_y + (obb_h/2) - (w+h)/2))
    center_offset = np.matmul(center_offset, cv2.getRotationMatrix2D((0,0), angle, 1)[:,0:2])
    rect_center_x = center_x + center_offset[0]
    rect_center_y = center_y + center_offset[1]
    obb = ((rect_center_x, rect_center_y), (obb_w, obb_h), angle)
    return obb

def calc_wl_from_mask(mask):
    contours,_ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    # there might be multiple contours, but we only want the largest one
    if len(contours) > 1:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
    obb = oriented_bounding_box(contours[0])
    w,l = obb[1]
    return w,l

def convex_volume_from_mask(height_map, mm_per_pixel):
    # reformat the data from [[z,z],[z,z]] to [[x,y,z],[x,y,z]]
    nonzero_inds = np.argwhere(height_map > 0).astype(np.uint16)
    nonzero_heights = height_map[height_map > 0].astype(np.uint16)
    # you need to have points for both the top surface profile and the base
    # otherwise the hull volume will be underestimated
    top_3d_coords = np.column_stack((nonzero_inds,
                                        nonzero_heights))
    base_3d_coords = np.column_stack((nonzero_inds,
                                        [0] * nonzero_inds.shape[0]))
    points_3d = np.vstack((top_3d_coords, base_3d_coords))
    # now can fit a convex hull and pull volume from that
    tuber_hull_3d = ConvexHull(points_3d)
    hull_volume_cm3 =  tuber_hull_3d.volume * (mm_per_pixel ** 2) / 1000
    return hull_volume_cm3

def draw_obb_on_image(image, obb, color=(255,0,0), thickness=3):
    # Draws the oriented bounding box on the image
    box = cv2.boxPoints(obb)
    box = np.intp(box)
    annotated = image.copy()
    cv2.drawContours(annotated, [box], 0, color, thickness)
    return annotated

class SingleTuberImgPair():
    def __init__(self, res, annotate_dir=None):
        self.res = res.cpu()
        self.rgb_path = Path(res.path)
        print(self.rgb_path)
        # add the depth path, based on the rgb path
        parts = list(self.rgb_path.parts)
        if len(parts) > 2 and parts[-2] == "rgb":
            parts[-2] = "depth"
        self.depth_path = Path(*parts)
        self.rgb_png = skimage.io.imread(str(self.rgb_path))
        self.depth_png = skimage.io.imread(str(self.depth_path))
        self.tuber_metrics = {}
        self.annotate_dir = annotate_dir
    def calc_cls_proportions(self):
        # calculate the proportion of each class in the masks
        np_mask = self.res.masks.data.numpy()
        cls_proportions = {cls:np.float32(0.0) for cls in self.res.names.values()}
        px_sum = np.sum(np_mask)
        # update with sums from the identified classes
        # note that multiple masks can have same class, use +=
        for i,cls in enumerate(self.res.boxes.cls):
            cls_proportions[self.res.names[int(cls)]] += np.sum(np_mask[i,:,:])/px_sum
        self.cls_proportions = cls_proportions

    def mask_fg(self):
        # create a binary mask of the foreground
        # this includes all classes except 'Sprout'
        nonsprout_keys = [k for k in self.res.names.keys() if self.res.names[k] != 'Sprout']
        # then get the indices of masks that correspond to these keys
        nonsprout_idx = [i for i,x in enumerate(self.res.boxes.cls.numpy()) if x in nonsprout_keys]
        nonsprout_masks = self.res.masks.data[nonsprout_idx,:,:]
        fg_mask = np.sum(nonsprout_masks.numpy(),0) > 0
        self.fg_mask = skimage.transform.resize(fg_mask, self.res.orig_shape)
        cv2.imwrite('fg_mask.png', (self.fg_mask * 255).astype(np.uint8))
        
    def calc_tuber_size(self, mm_per_pixel = 22.4/97):
        # using the mm per pixel value from the calibration run squares
        w,l = calc_wl_from_mask(self.fg_mask)
        self.tuber_metrics['length_cm'] = l * mm_per_pixel / 10
        self.tuber_metrics['width_cm'] = w * mm_per_pixel / 10
        height_mm = inpaint_to_height(self.depth_png)
        self.tuber_metrics['depth_cm'] = np.max(height_mm) / 10
        self.tuber_metrics['cross_sectional_area_cm2'] = np.sum(self.fg_mask) * (mm_per_pixel ** 2) / 100
        self.tuber_metrics['volume_cm3'] = np.sum(height_mm) * (mm_per_pixel ** 2) / 1000
        # need to zero out the background for the convex hull calculation
        background_zeroed = np.where(self.fg_mask, height_mm, 0)
        self.tuber_metrics['hull_volume_cm3'] = convex_volume_from_mask(background_zeroed, mm_per_pixel)
        self.tuber_metrics['solidity'] = self.tuber_metrics['volume_cm3'] / self.tuber_metrics['hull_volume_cm3'] 
        # Annotate and save image if requested
        if self.annotate_dir is not None:
            # Get OBB for drawing
            contours,_ = cv2.findContours(self.fg_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
            if len(contours) > 1:
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
            obb = oriented_bounding_box(contours[0])
            # Draw OBB on RGB image
            rgb_for_draw = self.rgb_png.copy()
            if rgb_for_draw.ndim == 2 or rgb_for_draw.shape[2] == 1:
                rgb_for_draw = cv2.cvtColor(rgb_for_draw, cv2.COLOR_GRAY2BGR)
            annotated = draw_obb_on_image(rgb_for_draw, obb, color=(255,0,0), thickness=3)
            out_path = Path(self.annotate_dir) / self.rgb_path.name
            cv2.imwrite(str(out_path), annotated[:,:,::-1])  # Convert BGR to RGB for saving

class TuberImgSet():
    def __init__(self, base_img_path, annotate_dir=None):
        self.base_img_path = Path(base_img_path)
        self.rgb_img_paths = sorted(self.base_img_path.glob('Tracked_Images/rgb/Img*.png'))
        self.depth_img_paths = sorted(self.base_img_path.glob('Tracked_Images/depth/Img*.png'))
        self.annotate_dir = annotate_dir
    def run_yolo_seg(self, model, verbose=False):
        print('Running YOLO11x-seg on dir: %s' % self.base_img_path)
        self.yolo_res = model.predict(self.base_img_path / 'Tracked_Images/rgb', verbose=verbose)
    def calc_all_metrics(self):
        print('Calculating metrics...')
        all_cls_props = []
        all_sizes = []
        all_imnames = []
        for res in self.yolo_res:
            img_pair = SingleTuberImgPair(res, annotate_dir=self.annotate_dir)
            if not img_pair.res.masks:
                continue
            img_pair.calc_cls_proportions()
            img_pair.mask_fg()
            img_pair.calc_tuber_size()
            all_cls_props.append(img_pair.cls_proportions)
            all_sizes.append(img_pair.tuber_metrics)
            all_imnames.append(img_pair.rgb_path.name)
        props = pd.DataFrame(all_cls_props)
        sizes = pd.DataFrame(all_sizes)
        df = props.join(sizes)
        df.insert(0,'dir',self.base_img_path)
        df.insert(1,'img',all_imnames)
        self.df = df
    def append_notes(self):
        col_headers = ["Type", "Knobs", "Tuber bend", "Sprouting", "Growth cracks", "Eye depth", "Eye number", "Russeting", "Aligator hide", "Skin defect", "Greening"]
        for header in col_headers:
            self.df[header] = ""  # Initialize columns with empty strings
        notes_path = self.base_img_path / 'notes.txt'
        if not notes_path.exists():
            print(f"Notes file {notes_path} does not exist, skipping notes for this dir.")
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
    annotate_dir = args.annotate_dir
    for target_subdir in target_subdirs:
        if annotate_dir is not None:
            # Create output dir if it doesn't exist
            Path(annotate_dir).mkdir(parents=True, exist_ok=True)
        img_set = TuberImgSet(target_subdir, annotate_dir=annotate_dir)
        if len(img_set.rgb_img_paths) != len(img_set.depth_img_paths):
            print(f"Skipping {target_subdir} due to mismatched RGB and depth image counts.")
            print(f"Found {len(img_set.rgb_img_paths)} RGB images and {len(img_set.depth_img_paths)} depth images.")
            continue
        if len(img_set.rgb_img_paths) == 0 or len(img_set.depth_img_paths) == 0:
            print(f"Skipping {target_subdir} due to missing RGB or depth images.")
            continue
        img_set.run_yolo_seg(model)
        img_set.calc_all_metrics()
        img_set.append_notes()
        all_dfs.append(img_set.df)
        out_df = pd.concat(all_dfs, ignore_index=True)
        out_df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
