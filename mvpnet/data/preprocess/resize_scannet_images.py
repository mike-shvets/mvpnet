import argparse
import os
import os.path as osp
import glob
import natsort
from PIL import Image
import numpy as np
import multiprocessing as mp

resize = (160, 120)
# resize = (640, 480)
# adapt the following paths

parser = argparse.ArgumentParser("Paths args")
parser.add_argument("scannet_root", help="Path to scannet root folder.")
parser.add_argument("--label-mode", default="label", choices=["label", "label-filt"],
                    help="Specificator of which label files to read.")
parser.add_argument("--num-processes", default=16, type=int,
                    help="Number of parallel processes.")
args = parser.parse_args()

raw_dir = os.path.join(args.scannet_root, "scans")
assert os.path.exists(raw_dir), "scans folder not found under the provided scannet root."
out_dir = os.path.join(args.scannet_root, "scans_resize_{}x{}".format(resize[0], resize[1]))

exclude_frames = {
    'scene0243_00': ['1175', '1176', '1177', '1178', '1179', '1180', '1181', '1182', '1183', '1184'],
    'scene0538_00': ['1925', '1928', '1929', '1931', '1932', '1933'],
    'scene0639_00': ['442', '443', '444'],
    'scene0299_01': ['1512'],
}


def worker_func(scan_id):
    scan_dir = osp.join(raw_dir, scan_id)
    color_path = osp.join(scan_dir, 'color', '{}.jpg')
    label_path = osp.join(scan_dir, args.label_mode, '{}.png')
    depth_path = osp.join(scan_dir, 'depth', '{}.png')

    # get all frame ids
    color_paths = natsort.natsorted(glob.glob(osp.join(scan_dir, 'color', '*.jpg')))
    exclude_ids = exclude_frames.get(scan_id, [])
    frame_ids = [osp.splitext(osp.basename(x))[0] for x in color_paths]
    frame_ids = [x for x in frame_ids if x not in exclude_ids]

    # resize
    for frame_id in frame_ids:
        color = Image.open(color_path.format(frame_id))
        label = Image.open(label_path.format(frame_id))
        depth = Image.open(depth_path.format(frame_id))

        if resize != color.size:
            color = color.resize(resize, Image.BILINEAR)
        if resize != label.size:
            label = label.resize(resize, Image.NEAREST)
        if resize != depth.size:
            depth = depth.resize(resize, Image.NEAREST)

        save_dict = {
            'color': color,
            args.label_mode: label,
            'depth': depth,
        }
        for k, img in save_dict.items():
            save_dir = osp.join(out_dir, scan_id, k)
            save_path = osp.join(save_dir, '{}.png'.format(frame_id))
            if not osp.exists(save_dir):
                os.makedirs(save_dir)
            img.save(save_path)
    print(scan_id)


# main
if not osp.exists(out_dir):
    os.makedirs(out_dir)
scan_ids = sorted(os.listdir(raw_dir))
# for scan_id in scan_ids:
#     worker_func(scan_id)
num_processes = min(args.num_processes, mp.cpu_count())
num_processes = min(num_processes, len(scan_ids))
p = mp.Pool(processes=num_processes)
p.map(worker_func, scan_ids, chunksize=1)
p.close()
p.join()

