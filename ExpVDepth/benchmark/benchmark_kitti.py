import os, sys, argparse
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, project_root)
import cv2
import einops
import numpy as np
import PIL.Image as Image

from tqdm import tqdm
from tabulate import tabulate
from matplotlib import pyplot as plt
import torch
import torch.utils.data as data
from utils.utils import InputPadder, to_cuda, readFlowKITTI, writeFlowKITTI
from utils.evaluation import compute_errors, upgrade_measures
from utils.pose_estimator import PoseEstimator

from ExpVDepth.RAFT.raft import RAFT
from ExpVDepth.LDNet.LightedDepthNet import LightedDepthNet
from ExpVDepth.datasets.kitti_eigen_test import KITTI_eigen
def is_valid_pose(pose):
    # KITTI dataset assumes camera always move forwards
    pose = pose.squeeze()
    R, t = pose[0:3, 0:3], pose[0:3, 3:4]
    if R[0, 0] < 0 or R[1, 1] < 0 or R[2, 2] < 0 or t[2] > 0:
        return False
    else:
        return True

def read_splits_kitti_test():
    split_root = os.path.join(project_root, 'misc', 'kitti_splits')
    entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'test_copy.txt'), 'r')]
    return entries

@torch.no_grad()
def validate_kitti(raft, lighteddepth, args, iters=24):
    """
    Validation on Kitti Eigen Split
    scale_th: even scale_th is 0.0, it may still run over frames, as algorithm automatically seek next frame when pose esitmation fail
    """
    assert args.scale_th >= 0.0

    raft.eval()
    lighteddepth.eval()

    pose_estimator = PoseEstimator(npts=10000, device=torch.device("cuda:6"))

    val_dataset = KITTI_eigen(
        data_root=args.kitti_root,
        entries=read_splits_kitti_test(),
        net_ht=args.net_ht, net_wd=args.net_wd,
        mono_depth_root=os.path.join(args.est_inputs_root, 'monodepth_kitti', args.mono_depth_method),
        grdt_depth_root=args.grdt_depth_root
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False
    )

    vmeasures, vmeasures_sfm, mmeasures, mmeasures_sfm = np.zeros(11), np.zeros(11), np.zeros(11), np.zeros(11)

    for val_id, data_blob in enumerate(tqdm(val_loader)):
        data_blob = to_cuda(data_blob)

        seq, frmidx, _ = data_blob['tag'][0].split(' ')
        h, w = data_blob['uncropped_size'][0]

        frmidx, h, w = int(frmidx), h.item(), w.item()

        # Lighted Depth Takes Image of size 320 x 1216
        crph, crpw = 320, 1216
        left, top = int((w - crpw) / 2), int(h - crph)

        eigen_crop_mask = np.ones([int(h), int(w)])
        eigen_crop_mask[int(0.3324324 * h):int(0.91351351 * h), int(0.0359477 * w):int(0.96405229 * w)] = 1
        eigen_crop_mask = eigen_crop_mask[top:top + crph, left:left + crpw]
        eigen_crop_mask = eigen_crop_mask == 1

        # Pose Estimation Stage
        coords1_init, interval = None, 1

        while(True):
            img1_idx = frmidx
            img2_idx = frmidx + interval

            img1path = os.path.join(
                args.kitti_root, seq, 'image_02/data', '{}.png'.format(str(img1_idx).zfill(10))
            )
            img2path = os.path.join(
                args.kitti_root, seq, 'image_02/data', '{}.png'.format(str(img2_idx).zfill(10))
            )

            if not os.path.exists(img2path):
                img2path = os.path.join(
                    args.kitti_root, seq, 'image_02/data', '{}.png'.format(str(frmidx + interval - 1).zfill(10))
                )
                break

            image1 = torch.from_numpy(np.array(Image.open(img1path))).permute([2, 0, 1]).float()[None].cuda(6)
            image2 = torch.from_numpy(np.array(Image.open(img2path))).permute([2, 0, 1]).float()[None].cuda(6)

            padder = InputPadder(image1.shape, mode='kitti')
            image1, image2 = padder.pad(image1, image2)

            flow_root = os.path.join(args.est_inputs_root, 'opticalflow_kitti', args.flow_init, seq)
            flow_path = os.path.join(flow_root, '{}_{}.png'.format(str(img1_idx).zfill(10), str(img2_idx).zfill(10)))
            if not os.path.exists(flow_path):
                os.makedirs(flow_root, exist_ok=True)
                flow, coords1_init = raft(image1, image2, iters=iters, coords1_init=coords1_init)
                flow = padder.unpad(flow)
                flow = einops.rearrange(flow.squeeze(), 'd h w -> h w d')

                flownp = flow.cpu().numpy()
                writeFlowKITTI(flow_path, flownp)
            else:
                flownp, _ = readFlowKITTI(flow_path)
                flow = torch.from_numpy(flownp).float().cuda(6)

            mdepth_uncropped, intrinsic_uncropped = data_blob['mono_depth_uncropped'].squeeze(), data_blob['intrinsic_uncropped'].squeeze()
            valid_regeion = torch.zeros([h, w], device=mdepth_uncropped.device, dtype=torch.bool)
            valid_regeion[int(0.25810811 * h):int(0.99189189 * h)] = 1
            pose, scale_md = pose_estimator.pose_estimation(flow, mdepth_uncropped, intrinsic_uncropped[0:3, 0:3], valid_regeion=valid_regeion, seed=val_id)
            if is_valid_pose(pose):
                break
            else:
                interval += 1

        image1 = torch.from_numpy(np.array(Image.open(img1path))).permute([2, 0, 1]).float().unsqueeze(0).cuda(6)
        image2 = torch.from_numpy(np.array(Image.open(img2path))).permute([2, 0, 1]).float().unsqueeze(0).cuda(6)

        image1 = image1[:, :, top:top + crph, left:left + crpw].contiguous() / 255.0
        image2 = image2[:, :, top:top + crph, left:left + crpw].contiguous() / 255.0

        mdepth_cropped = data_blob['mono_depth_cropped']
        mdepth_cropped = torch.clamp_min(mdepth_cropped, min=args.min_depth_pred)

        intrinsic_cropped = data_blob['intrinsic_cropped']

        outputs = lighteddepth(image1, image2, mdepth_cropped, intrinsic_cropped, pose)

        gtdepth = data_blob['grdt_depth_cropped'].squeeze().cpu().numpy()
        vdepth = outputs[('depth', 1)].squeeze().cpu().numpy()
        mdepth = data_blob['mono_depth_cropped'].squeeze().cpu().numpy()
        selector = (gtdepth > 1e-3) * (gtdepth < 80) * eigen_crop_mask
        
        mmeasures = upgrade_measures(mdepth, gtdepth, selector, mmeasures, SfM=False)
        mmeasures_sfm = upgrade_measures(mdepth, gtdepth, selector, mmeasures_sfm, SfM=True)
        vmeasures = upgrade_measures(vdepth, gtdepth, selector, vmeasures, SfM=False)
        vmeasures_sfm = upgrade_measures(vdepth, gtdepth, selector, vmeasures_sfm, SfM=True)

        # min_d = vdepth.min()
        # max_d = vdepth.max()
        # vdepth = (vdepth - min_d) / (max_d - min_d)
        # vdepth = vdepth.clip(0,1)
        # depth_colored = colorize_depth_maps(vdepth,0,1,cmap='Spectral').squeeze()
        # depth_colored = (depth_colored * 255).astype(np.uint8)
        # depth_colored_hwc = chw2hwc(depth_colored)
        # depth_colored_img = Image.fromarray(depth_colored_hwc)
        # depth_colored_img.save(os.path.join(args.output_root,'{}.png'.format(str(frmidx).zfill(10))))
        plt.imsave(os.path.join(args.output_rgb_root,'{}.png'.format(str(frmidx).zfill(10))), vdepth, cmap='Spectral',vmin=vdepth.min(),vmax=vdepth.max())
        
        min_d = vdepth.min()
        max_d = vdepth.max()
        vdepth = 65535.0 * (vdepth - min_d) / (max_d - min_d)
        cv2.imwrite(os.path.join(args.output_gray_root,'{}.png'.format(str(frmidx).zfill(10))), vdepth.astype("uint16"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
    mmeasures[0:10] = mmeasures[0:10] / mmeasures[10]
    mmeasures_sfm[0:10] = mmeasures_sfm[0:10] / mmeasures_sfm[10]
    vmeasures[0:10] = vmeasures[0:10] / vmeasures[10]
    vmeasures_sfm[0:10] = vmeasures_sfm[0:10] / vmeasures_sfm[10]

    print('KITTI Performance Reported over %f eval samples:' % (vmeasures[10].item()))
    table = [
        ['', 'ScInv', 'log10', 'silog', 'abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1', 'd2', 'd3'],
        ['MDepth'] + list(mmeasures[0:10]),
        ['MDepth-SfM'] + list(mmeasures_sfm[0:10]),
        ['VDepth'] + list(vmeasures[0:10]),
        ['VDepth-SfM'] + list(vmeasures_sfm[0:10]),
    ]
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid', numalign="center", floatfmt=".3f"))
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raft_ckpt',                  help="checkpoint of RAFT",                      type=str,   default='/home/zhujiajun/LightedDepth-main/misc/checkpoints/raft-sintel.pth')
    parser.add_argument('--ldepth_ckpt',                help="checkpoint of LightedDepth",              type=str,   default='/home/zhujiajun/LightedDepth-main/misc/checkpoints/lighteddepth_kitti.pth')

    parser.add_argument('--kitti_root',                 help="root of kitti",                           type=str,   default="/data/dataset/kitti/raw_data/")
    parser.add_argument('--grdt_depth_root',            help="root of grdt_depth",                      type=str,   default="/data/dataset/kitti/data_depth_annotated/val/2011_09_26_drive_0013_sync/proj_depth/groundtruth/")
    parser.add_argument('--est_inputs_root',            help="root of estimated_inputs",                type=str,   default=os.path.join(project_root, 'estimated_inputs'))
    parser.add_argument('--mono_depth_method',          help="method name of mono_depth",               type=str,   default='marigold')
    parser.add_argument('--flow_init',                  help="method name of optical flow",             type=str,   default='raft')
    parser.add_argument('--output_rgb_root',            help="root of RGB output directory",            type=str,   default=os.path.join(project_root, 'output0325/marigold/rgb'))
    parser.add_argument('--output_gray_root',           help="root of Gray output directory",           type=str,   default=os.path.join(project_root, 'output0325/marigold/gray'))
    parser.add_argument('--net_ht',                                                                     type=int,   default=320)
    parser.add_argument('--net_wd',                                                                     type=int,   default=1216)
    parser.add_argument('--min_depth_pred',             help="minimal evaluate depth",                  type=float, default=1)
    parser.add_argument('--maxlogscale',                help="maximum stereo residual value",           type=float, default=1.5)
    parser.add_argument('--scale_th',                   help="minimal camera translation magnitude",    type=float, default=0.0)
    parser.add_argument('--dataset_type',               help="which experiment dataset",                type=str,   default="kitti")

    args = parser.parse_args()

    assert args.mono_depth_method in ['bts', 'adabins', 'newcrfs', 'monodepth2','marigold']

    if not os.path.exists(args.output_rgb_root):
        os.makedirs(args.output_rgb_root, exist_ok=True)
        
    if not os.path.exists(args.output_gray_root):
        os.makedirs(args.output_gray_root, exist_ok=True)
    
    raft = RAFT(args)
    raft.load_state_dict(torch.load(args.raft_ckpt, map_location="cpu"), strict=True)
    raft.cuda(6)
    raft.eval()

    lighteddepth = LightedDepthNet(args=args)
    lighteddepth.load_state_dict(torch.load(args.ldepth_ckpt, map_location="cpu"), strict=True)
    lighteddepth.cuda(6)
    lighteddepth.eval()

    validate_kitti(raft, lighteddepth, args)