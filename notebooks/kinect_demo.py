import marimo

__generated_with = "0.10.16"
app = marimo.App()


@app.cell
def _():
    # demo pipeline of measuring 3d solidity off Kinect images

    import skimage
    import re
    import pandas as pd
    import numpy as np
    from skimage.restoration import inpaint
    from skimage.filters import threshold_mean
    from scipy.spatial import ConvexHull
    from pathlib import Path
    return ConvexHull, Path, inpaint, np, pd, re, skimage, threshold_mean


@app.cell
def _(ConvexHull, inpaint, np, re, skimage, threshold_mean):
    # inpaint and invert from depth to height
    def sk_inpaint_to_height(img):
        mask = img == 0
        # skimage will mess up uint16 input for some reason
        # convert to float64 then back to uint16
        depth_mm = inpaint.inpaint_biharmonic(img.astype(np.float64), mask).astype(np.uint16)
        height_mm = np.max(depth_mm) - depth_mm
        return height_mm

    def scale_height_to_8bit(img):
        out = 255.0 * (img - np.min(img))/(np.max(img) - np.min(img))
        out = out.astype(np.uint8)
        return out

    # take a depth PNG path and return volumes and solidity
    def solidity_3d(path):
        depth = skimage.io.imread(str(path))
        rgb = skimage.io.imread(re.sub('depth','rgb',str(path)))
        hsv = skimage.color.rgb2hsv(rgb)
        # add a couple thresholds
        height_scaled = sk_inpaint_to_height(depth)
        tall_fg = height_scaled > threshold_mean(height_scaled)
        dist_from_blue = np.abs(hsv[:,:,0] - 0.6)
        nonblue_fg = dist_from_blue > threshold_mean(dist_from_blue)
        # zero out the BG for final-ish image
        bg_zeroed = np.where(nonblue_fg, height_scaled, 0)
        # get the volume itself
        # using a very rough guess for mm_per_pixel here
        mm_per_pixel = 304/1750
        volume_cm3 = np.sum(bg_zeroed) * (mm_per_pixel ** 2) / 1000
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
        hull_volume_cm3 = tuber_hull_3d.volume * (mm_per_pixel ** 2) / 1000
        solidity_3d = volume_cm3 / hull_volume_cm3
        return volume_cm3, hull_volume_cm3, solidity_3d, bg_zeroed
    return scale_height_to_8bit, sk_inpaint_to_height, solidity_3d


@app.cell
def _(Path, scale_height_to_8bit, skimage, solidity_3d):
    target_dir = Path('sample_images/O23-0765/')
    annot_dir = Path('depth_maps_O23-0765/')

    annot_dir.mkdir()

    all_results = []
    for impath in target_dir.glob('Tracked_Images/depth/*.png'):
        print(impath)
        outlist = [target_dir.name, impath.name]
        vol, hull, solidity, bg_zeroed = solidity_3d(impath)
        outlist += [vol, hull, solidity]
        outlist.append(1-solidity)
        outlist.append(str(impath))
        skimage.io.imsave(str(annot_dir / Path(impath.name)),
                          scale_height_to_8bit(bg_zeroed))
        all_results.append(outlist)
    return (
        all_results,
        annot_dir,
        bg_zeroed,
        hull,
        impath,
        outlist,
        solidity,
        target_dir,
        vol,
    )


@app.cell
def _(all_results, pd):
    res_df = pd.DataFrame(all_results, columns=['variety','img_name','est_tuber_vol_cm3','est_convex_vol_cm3','solidity','knobbiness','img_path'])

    res_df.sort_values(by = ['solidity'], inplace=True)
    res_df.to_csv('solidity_O23-0765.csv', index=False)

    # notes as of 10-14-24
    # the solidity works alright, but we really need tuber outlines to
    return (res_df,)


if __name__ == "__main__":
    app.run()
