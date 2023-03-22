import os.path
import pickle
import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rotation
import joblib
import cv2
import trimesh
from pytorch3d import transforms
import torch.nn.functional as F


AHOI_ROOT = './data/AHOI_ROOT'
IMAGE_FOLDER = './data/IMG_FOLDER'
SMPL_MODEL_FOLDER = './data/body_models'
DATA_FOLDER = './data/DATA_FOLDER'
MODEL_FOLDER = './data/checkpoints'
OUTPUT_DIR = './data/outputs'


PART_LIST = ['chair_head', 'chair_back', 'chair_arm_left', 'chair_arm_right', 'chair_seat', 'chair_base', 'footrest']
ITEM_LIST = ['object_location',
             'object_rotation',
             'object_root_location',
             'object_root_rotation',
             'object_id.npy',
             'human_pose.npy',
             'human_orient.npy',
             'human_transl.npy',
             'human_betas.npy',
             'img_name.npy',
             'joint_prox.npy']
PARENT_DICT = {4: None, 1: 4, 2: 4, 3: 4, 5: 4, 6: 4, 0: 1}


def load_data(filepath):
    ext = filepath.split('.')[-1].lower()
    data = None
    if ext == 'pkl':
        with open(filepath, 'rb') as f:
            data = joblib.load(f)
    elif ext == 'json':
        with open(filepath, 'r') as f:
            data = json.load(f)
    elif ext == 'npy':
        try:
            data = np.load(filepath, allow_pickle=True).item()
        except:
            data = np.load(filepath, allow_pickle=True)

    return data


def create_mat(rot, transl, rot_type='matrix', affine=False):
    if affine:
        rot = [rot[2], rot[1], rot[0]]
        transl = [-transl[2], -transl[1], -transl[0]]
    if rot_type =='rot_vec':
        rot = Rotation.from_rotvec(rot).as_matrix()
    elif rot_type != 'matrix':
        rot = Rotation.from_euler(rot_type, rot).as_matrix()
    if affine:
        R, T = np.eye(4), np.eye(4)
        R[:3, :3] = rot
        T[:3, 3] = transl
        mat = R @ T
    else:
        mat = np.eye(4)
        mat[:3, :3] = rot
        mat[:3, 3] = transl
    return mat.astype(np.float32)

def create_mat_torch(rot, transl, rot_type='matrix', affine=False):
    if affine:
        rot = torch.tensor([[1., 0., 0], [0., 1., 0], [0., 0., 1.]]) @ rot.view(3, 1)
        transl = torch.tensor([[0., 0., -1.], [0., -1., 0.], [-1., 0., 0.]]) @ transl.view(3, 1)
        rot = rot.view((-1,))
        transl = transl.view((-1,))
    else:
        rot = torch.tensor([rot[2], rot[1], rot[0]])
        transl = torch.tensor([transl[0], transl[1], transl[2]])
    if rot_type =='rot_vec':
        rot = transforms.axis_angle_to_matrix(rot)
    elif rot_type != 'matrix':
        rot_type = rot_type[::-1]
        rot = transforms.euler_angles_to_matrix(rot, rot_type)
    if affine:
        R, T = torch.eye(4), torch.eye(4)
        R[:3, :3] = rot
        T[:3, 3] = transl
        mat = R @ T
    else:
        mat = torch.eye(4)
        mat[:3, :3] = rot
        mat[:3, 3] = transl
    return mat.to(torch.float32)

def create_mat_batch(rot, transl, rot_type='matrix', affine=False):
    if isinstance(rot, list):
        rot = np.stack(rot)
    if isinstance(transl, list):
        transl = np.stack(transl)
    rot[np.isnan(rot)] = 0
    transl[np.isnan(transl)] = 0
    batch = len(rot)
    if affine:
        rot = np.concatenate([rot[:, [2]], rot[:, [1]], rot[:, [0]]], axis=1)
        transl = np.concatenate([-transl[:, [2]], -transl[:, [1]], -transl[:, [0]]], axis=1)
    if rot_type=='rot_vec':
        rot = Rotation.from_rotvec(rot).as_matrix()
    elif rot_type != 'matrix':
        rot = Rotation.from_euler(rot_type, rot).as_matrix()
    if affine:
        R, T = np.eye(4), np.eye(4)
        R = np.repeat(R[None, ...], batch, axis=0)
        T = np.repeat(T[None, ...], batch, axis=0)
        R[:, :3, :3] = rot
        T[:, :3, 3] = transl

        mat = np.matmul(R, T)
    else:
        mat = np.eye(4)
        mat = np.repeat(mat[None, ...], batch, axis=0)
        mat[:, :3, :3] = rot
        mat[:, :3, 3] = transl
    return mat.astype(np.float32)


def voxel_to_pcd(grid, voxel):
    occ_bin = voxel.clone()
    occ_bin[occ_bin >= 0.3] = 1.
    occ_bin[occ_bin < 0.3] = 0.
    occ_bin = occ_bin.reshape(-1).to(torch.bool).detach().cpu()
    grid = grid.squeeze()
    pcd = grid[occ_bin]
    pcd_trimesh = trimesh.points.PointCloud(vertices=pcd)

    return pcd_trimesh


def mat_to_rot_loc(mat):
    rot = Rotation.from_matrix(mat[:3, :3]).as_euler('xyz')
    loc = mat[:3, 3]
    return rot, loc


def load_pose_data(video_id, view_id=0):
    pose_file = os.path.join(AHOI_ROOT, "Poses", f'{video_id}_{view_id}.pkl')
    if not os.path.exists(pose_file):
        return None
    with open(pose_file, 'rb') as f:
        data = pickle.load(f)
    return data

def load_obj_meta(array=False):
    if not array:
        meta_file = os.path.join(AHOI_ROOT, 'Metas', 'object_meta.pkl')
        with open(meta_file, 'rb') as f:
            data = pickle.load(f)
        return data
    else:
        meta_file = os.path.join(AHOI_ROOT, 'Metas', 'object_info.npy')
        data = np.load(meta_file, allow_pickle=True).item()
        return data

def load_obj_voxel(id, n_grid=128):
    if n_grid == 128:
        filepath = os.path.join(AHOI_ROOT, 'object_part_voxel', str(id) + '.npy')
    elif n_grid == 64:
        filepath = os.path.join(AHOI_ROOT, 'object_part_voxel_64', str(id) + '.npy')
    else:
        raise NotImplementedError
    voxel = np.load(filepath)
    return voxel


def select_nth_data(data, ind):
    data_out = {}
    for key in data.keys():
        data_out[key] = data[key][ind]
    return  data_out


def rectify_pose(pose):
    """
    Rectify "upside down" people in global coord

    Args:
        pose (72,): Pose.

    Returns:
        Rotated pose.
    """
    pose = pose.copy()
    R_mod = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
    R_root = cv2.Rodrigues(pose[:3])[0]
    new_root = R_root.dot(R_mod)
    pose[:3] = cv2.Rodrigues(new_root)[0].reshape(3)
    return pose

def gene_voxel_grid(N=64, len=1, homo=True):
    x_ = np.linspace(-len / 2., len / 2., N)
    y_ = np.linspace(-len / 2., len / 2., N)
    z_ = np.linspace(-len / 2., len / 2., N)

    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')

    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    if homo:
        d = np.ones_like(z)
        mg = np.stack([x, y, z, d]).T.astype(np.float32)
    else:
        mg = np.stack([x, y, z]).T.astype(np.float32)

    return mg

def load_mesh(object_meta, object_id, part_id, trans_M, as_pcd=False, to_trimesh=False):
    object_ind_in_meta = np.where(object_meta['object_ids'] == int(object_id))[0][0]
    face_len = object_meta['face_len'][object_ind_in_meta][part_id]
    if not face_len:
        if to_trimesh:
            return None
        else:
            return None, None
    vert_len = object_meta['vertex_len'][object_ind_in_meta][part_id]

    if to_trimesh:
        vertices = object_meta['vertices'][object_ind_in_meta][part_id][:vert_len, :3]
        faces = object_meta['faces'][object_ind_in_meta][part_id][:face_len]
        part_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        part_mesh.apply_transform(trans_M)
        return part_mesh
    else:
        vertices = object_meta['vertices'][object_ind_in_meta][part_id][:vert_len] @ trans_M.T
        vertices = vertices[:, :3]
        faces = object_meta['faces'][object_ind_in_meta][part_id][:face_len]
        return vertices, faces


def trans_pcd(pcd, trans_M):
    d = np.ones((pcd.shape[0], 1))
    pcd = np.concatenate([pcd, d], axis=1)
    pcd = pcd @ trans_M.T
    pcd = pcd[:, :3]

    return pcd


def apply_part_transform(rot_param, transl_param, global_mat, object_meta, voxels):
    mat_to_parent = torch.eye(4).repeat(7, 1, 1)
    init_shift = torch.from_numpy(object_meta['init_shift'])
    part_rot = torch.from_numpy(object_meta['part_transform'])
    for pid, parent_id in PARENT_DICT.items():
        if pid == 4 or not torch.any(init_shift[pid]):
            continue
        shift_to_parent = init_shift[pid] - init_shift[parent_id]
        if part_rot[pid][0]:
            shift_to_joint = -part_rot[pid][2:]
            mat_to_joint = create_mat_torch(torch.tensor([0., 0., 0.]), shift_to_joint, rot_type='XYZ', affine=True)
            joint_rot = torch.tensor([0., 0., 0.])
            joint_rot[int(part_rot[pid][0] - 1)] = rot_param[pid]
            joint_rot_mat = create_mat_torch(joint_rot, torch.tensor([0., 0., 0.]), rot_type='XYZ', affine=True)
            # joint_rot_mat = torch.linalg.inv(mat_to_joint) @ joint_rot_mat @ mat_to_joint
            joint_rot_mat = mat_to_joint @ joint_rot_mat @ torch.linalg.inv(mat_to_joint)
        else:
            joint_rot_mat = torch.eye(4)
        if part_rot[pid][1]:
            joint_shift = torch.tensor([0., 0., 0.])
            joint_shift[int(part_rot[pid][1] - 1)] = transl_param[pid]
            joint_shift_mat = create_mat_torch(torch.tensor([0., 0., 0.]), joint_shift, rot_type='XYZ', affine=True)
        else:
            joint_shift_mat = torch.eye(4)

        mat_to_parent[pid] = joint_rot_mat @ joint_shift_mat @ create_mat_torch(torch.tensor([0., 0., 0.]), shift_to_parent, rot_type='XYZ', affine=True)
    seat_to_world = create_mat_torch(torch.tensor([0., 0., 0.]), -init_shift[4], rot_type='XYZ') @ global_mat
    back_to_world = mat_to_parent[1] @ seat_to_world if torch.any(init_shift[1]) else torch.eye(4)
    head_to_world = mat_to_parent[0] @ back_to_world if torch.any(init_shift[0]) else torch.eye(4)
    leftarm_to_world = mat_to_parent[2] @ seat_to_world if torch.any(init_shift[2]) else torch.eye(4)
    rightarm_to_world = mat_to_parent[3] @ seat_to_world if torch.any(init_shift[3]) else torch.eye(4)
    base_to_world = mat_to_parent[5] @ seat_to_world if torch.any(init_shift[5]) else torch.eye(4)
    footrest_to_world = mat_to_parent[6] @ seat_to_world if torch.any(init_shift[6]) else torch.eye(4)
    mat_to_world = torch.stack([head_to_world, back_to_world, leftarm_to_world, rightarm_to_world,
                              seat_to_world, base_to_world, footrest_to_world], dim=0)

    mat_to_world = mat_to_world[:, :3, :]

    n_grid = voxels.shape[2]
    grid_affine = F.affine_grid(mat_to_world, size=[7, 1, n_grid, n_grid, n_grid], align_corners=False)
    voxel_affine = F.grid_sample(voxels, grid_affine, align_corners=False).view(7, n_grid, n_grid, n_grid)
    voxel_all = torch.max(voxel_affine, dim=0)[0]

    return voxel_all

