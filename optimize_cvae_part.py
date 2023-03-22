#%%
import os
from models_cvae import CVAE
from visualize import *
import torch
from PIL import Image
from io import BytesIO
from dataloaders import AhoiDatasetGlobal
from torch.utils.data import DataLoader
from models_voxel_pred import GPose
from pytorch3d import transforms as T
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--save_image', action="store_true", default=False)
parser.add_argument('--n_step', type=int, default=20)
parser.add_argument('--output_dir', type=str, default=False)
parser.add_argument('--n_grid', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--prior_hidden', type=int, default=16)
parser.add_argument('--recon_hidden', type=int, default=16)

args = parser.parse_args()

def optimize(args):

    object_meta_all = load_obj_meta(array=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    grid = gene_voxel_grid(N=args.n_grid, len=2, homo=False)
    grid = torch.from_numpy(grid)

    out_conv_channels = 512

    learning_rate = args.lr

    model_cvae = CVAE(dim=args.n_grid, out_conv_channels=out_conv_channels, hidden_dim=args.prior_hidden).to(device)
    MODEL_PATH = os.path.join(MODEL_FOLDER, 'cvae.pth')
    model_cvae.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model_cvae.eval()

    model_vox = GPose().to(device)
    MODEL_PATH = os.path.join(MODEL_FOLDER, 'model_voxel_pred.pth')
    model_vox.load_state_dict(torch.load(MODEL_PATH))
    model_vox.eval()

    train_dataset = AhoiDatasetGlobal(data_folder=DATA_FOLDER, n_grid=args.n_grid, add_human=True)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)

    global_ind = 0

    for i, (output, idx) in enumerate(train_loader):
        pelvis_transform = output['pelvis_transform'][0].detach().cpu().numpy()
        pelvis_transform_pare = output['pelvis_transform_pare'][0].detach().cpu().numpy()
        occ_human = output['occ_human']
        occ_pare_human = output['occ_pare_human']
        occ_object = output['occ']
        pare_human_pose = output['pare_human_pose']
        img = output['img']

        pcd_human = grid[occ_human.reshape(-1) > 0]
        pcd_pare_human = grid[occ_pare_human.reshape(-1) > 0]
        pcd_object = grid[occ_object.reshape(-1) > 0]

        object_id = output['object_id'][0].item()
        object_meta = object_meta_all[object_id]
        voxels = load_obj_voxel(object_id, n_grid=args.n_grid)
        voxels = torch.from_numpy(voxels).unsqueeze(1).to(torch.float32)

        with torch.no_grad():
            out_rot, out_loc = model_vox(pare_human_pose)
            out_rot_matrix = T.rotation_6d_to_matrix(out_rot)
            out_rot = T.matrix_to_euler_angles(out_rot_matrix, convention='XYZ')

        out_rot = torch.nn.Parameter(out_rot, requires_grad=True)
        out_loc = torch.nn.Parameter(out_loc, requires_grad=True)
        rot_param = torch.nn.Parameter(torch.zeros(7), requires_grad=True)
        transl_param = torch.nn.Parameter(torch.zeros(7), requires_grad=True)

        optimizer_rot = torch.optim.SGD([out_rot], lr=learning_rate * 3., momentum=0.2)
        optimizer_loc = torch.optim.SGD([out_loc], lr=learning_rate, momentum=0.2)
        optimizer_rot_param = torch.optim.SGD([rot_param], lr=learning_rate * 3., momentum=0.2)
        optimizer_transl_param = torch.optim.SGD([transl_param], lr=learning_rate, momentum=0.2)


        images = []
        for step in range(args.n_step + 1):
            out_mat = create_mat_torch(out_rot, out_loc, rot_type='XYZ', affine=True)

            voxel_all = apply_part_transform(rot_param, transl_param, out_mat, object_meta, voxels)

            x = voxel_all[None, None, ...]
            x = x.to(device)
            z, _ = model_cvae.encoder(x, occ_pare_human)

            with torch.no_grad():

                if step == 0 or step == args.n_step or args.save_image:

                    occ_obj = x.clone()
                    occ_obj[occ_obj >= 0.5] = 1
                    occ_obj[occ_obj < 0.5] = 0

                    iou = torch.count_nonzero(torch.logical_and(occ_obj, occ_object)) / torch.count_nonzero(torch.logical_or(occ_obj, occ_object))
                    print(f'IOU AT STEP {step}: {iou}')

                    occ_obj = x.view(args.n_grid ** 3)
                    occ_obj = occ_obj.to(torch.bool).detach().cpu().numpy()
                    pcd_object_pt = trimesh.points.PointCloud(vertices=grid[occ_obj])

                if args.save_image:
                    if step == args.n_step - 1:
                        pcd_pare_human = trans_pcd(pcd_pare_human, pelvis_transform_pare)
                        pcd_object = trans_pcd(pcd_object, pelvis_transform)
                        pcd_human_trimesh = trimesh.points.PointCloud(vertices=pcd_human)
                        pcd_object_trimesh = trimesh.points.PointCloud(vertices=pcd_object)

                        pcd_pt = pcd_object_pt.apply_transform(pelvis_transform_pare)
                        pcd_gt = pcd_object_trimesh
                        pcd_gt.visual.vertex_colors = [240, 20, 20]
                        pcd_gt[0].visual.vertex_colors = [255, 10, 10]
                        meshes = [pcd_pt] + [pcd_human_trimesh] + [pcd_gt]

                        img_bytes = visualize_result(show_axis=False, show=True, return_img=True, extra_meshes=meshes)
                    img = Image.open(BytesIO(img_bytes))
                    images.append(img)

            loss = z @ z.T #- pred_loss * 0.001
            optimizer_rot.zero_grad()
            optimizer_loc.zero_grad()
            optimizer_rot_param.zero_grad()
            optimizer_transl_param.zero_grad()
            loss.backward()

            optimizer_rot.step()
            optimizer_loc.step()
            optimizer_rot_param.step()
            optimizer_transl_param.step()

        if args.save_image:
            for _ in range(10):
                images.append(img)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            images[0].save(os.path.join(OUTPUT_DIR, f'vis_{global_ind:02d}.gif'), save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)

        global_ind += 1

        if global_ind >= 100:
            break




