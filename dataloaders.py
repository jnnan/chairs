from torch.utils.data import Dataset
from visualize import *
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms


train_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class AhoiDataset(Dataset):
    def __init__(self, data_folder, n_grid=64, len_grid=2, add_human=False, add_contrast=False):
        self.data = {}
        self.data['human_betas'] = load_data(os.path.join(data_folder, 'human_betas.npy'))
        self.data['object_location'] = load_data(os.path.join(data_folder, 'object_location.npy'))
        self.data['object_rotation'] = load_data(os.path.join(data_folder, 'object_rotation.npy'))
        self.data['object_id'] = load_data(os.path.join(data_folder, 'object_id.npy'))
        self.data['human_pose'] = load_data(os.path.join(data_folder, 'human_pose.npy'))
        self.grid = torch.from_numpy(gene_voxel_grid(N=n_grid, len=len_grid, homo=False))[None, :]
        self.n_grid = n_grid
        self.create_smplx_model()
        self.add_human = add_human
        self.add_contrast = add_contrast
        print('Loaded Data')

    def __len__(self):
        return len(self.data['object_id'])

    def __getitem__(self, idx):
        human_pose = self.data['human_pose'][idx].astype(np.float32)
        human_betas = self.data['human_betas'][idx].astype(np.float32)
        object_rot = self.data['object_rotation'][idx].astype(np.float32)
        object_loc = self.data['object_location'][idx].astype(np.float32)
        object_id = self.data['object_id'][idx]

        matrices = create_mat_batch(object_rot, object_loc, rot_type='xyz', affine=True)
        matrices = torch.from_numpy(matrices[:, :3, :]).to(torch.float32)

        voxels = load_obj_voxel(object_id, n_grid=self.n_grid)
        voxels = torch.from_numpy(voxels).unsqueeze(1).to(torch.float32)
        grid_affine = F.affine_grid(matrices, size=[7, 1, self.n_grid, self.n_grid, self.n_grid], align_corners=False)
        voxel_affine = F.grid_sample(voxels, grid_affine, align_corners=False)
        voxel_affine = voxel_affine.view(7, self.n_grid, self.n_grid, self.n_grid)
        voxel_all = torch.any(voxel_affine, dim=0)
        occ = voxel_all[None, ...].to(torch.float32)
        occ[occ < 0] = 0.
        occ[occ > 1] = 1.

        output = {}
        output['human_pose'] = human_pose
        output['human_betas'] = human_betas
        output['occ'] = occ

        output['object_id'] = object_id

        if self.add_contrast:

            matrices = create_mat_batch(object_rot + np.random.randn(3) * 0.1, object_loc + np.random.randn(3) * 0.2,
                                        rot_type='xyz', affine=True)
            matrices = torch.from_numpy(matrices[:, :3, :]).to(torch.float32)
            voxels = load_obj_voxel(object_id, n_grid=self.n_grid)
            voxels = torch.from_numpy(voxels).unsqueeze(1).to(torch.float32)
            grid_affine = F.affine_grid(matrices, size=[7, 1, self.n_grid, self.n_grid, self.n_grid], align_corners=False)
            voxel_affine = F.grid_sample(voxels, grid_affine, align_corners=False)
            voxel_affine = voxel_affine.view(7, self.n_grid, self.n_grid, self.n_grid)
            voxel_all = torch.any(voxel_affine, dim=0)
            occ = voxel_all[None, ...].to(torch.float32)
            occ[occ < 0] = 0.
            occ[occ > 1] = 1.
            output['occ_fake'] = occ

        if self.add_human:
            smplx_output = self.smplx_model(return_verts=True, body_pose=torch.tensor(human_pose[None, ...]),
                                            betas=torch.tensor(human_betas[None, ...]))
            vertices = smplx_output.vertices.detach().cpu().numpy().squeeze()
            joints = smplx_output.joints.detach().cpu().numpy().squeeze()
            pelvis_transform = create_mat([0, 0, 0], joints[0], rot_type='rot_vec') \
                               @ create_mat([0, np.pi, 0], np.array([0, 0, 0]), rot_type='xyz')
            vertices = trans_pcd(vertices, np.linalg.inv(pelvis_transform))
            faces = self.smplx_model.faces
            occ_human = voxelize_mesh(vertices, faces, self.grid)
            output['occ_human'] = occ_human[None, ...].to(torch.float32)

        return output, idx

    def create_smplx_model(self):
        self.smplx_model = smplx.create(MODEL_FOLDER, model_type='smplx',
                                   gender='male', ext='npz',
                                   num_betas=10,
                                   use_pca=False,
                                   create_global_orient=True,
                                   create_body_pose=True,
                                   create_betas=True,
                                   create_left_hand_pose=True,
                                   create_right_hand_pose=True,
                                   create_expression=True,
                                   create_jaw_pose=True,
                                   create_leye_pose=True,
                                   create_reye_pose=True,
                                   create_transl=True,
                                   batch_size=1)


class AhoiDatasetContrast(Dataset):
    def __init__(self, data_folder, n_grid=64, len_grid=2, add_human=False):
        self.data = {}
        self.data['human_betas'] = load_data(os.path.join(data_folder, 'human_betas.npy'))
        self.data['object_location'] = load_data(os.path.join(data_folder, 'object_location.npy'))
        self.data['object_rotation'] = load_data(os.path.join(data_folder, 'object_rotation.npy'))
        self.data['object_id'] = load_data(os.path.join(data_folder, 'object_id.npy'))
        self.data['human_pose'] = load_data(os.path.join(data_folder, 'human_pose.npy'))
        self.grid = torch.from_numpy(gene_voxel_grid(N=n_grid, len=len_grid, homo=False))[None, :]
        self.n_grid = n_grid
        self.create_smplx_model()
        self.add_human = add_human

        print('Loaded Data')

    def __len__(self):
        return len(self.data['object_id'])

    def __getitem__(self, idx):
        human_pose = self.data['human_pose'][idx].astype(np.float32)
        human_betas = self.data['human_betas'][idx].astype(np.float32)
        object_rot = self.data['object_rotation'][idx].astype(np.float32)
        object_loc = self.data['object_location'][idx].astype(np.float32)
        object_id = self.data['object_id'][idx]

        matrices = create_mat_batch(object_rot, object_loc, rot_type='xyz', affine=True)
        matrices = torch.from_numpy(matrices[:, :3, :]).to(torch.float32)
        voxels = load_obj_voxel(object_id, n_grid=self.n_grid)
        voxels = torch.from_numpy(voxels).unsqueeze(1).to(torch.float32)
        grid_affine = F.affine_grid(matrices, size=[7, 1, self.n_grid, self.n_grid, self.n_grid], align_corners=False)
        voxel_affine = F.grid_sample(voxels, grid_affine, align_corners=False)
        voxel_affine = voxel_affine.view(7, self.n_grid, self.n_grid, self.n_grid)
        voxel_all = torch.any(voxel_affine, dim=0)
        occ = voxel_all[None, ...].to(torch.float32)
        occ[occ < 0] = 0.
        occ[occ > 1] = 1.

        output = {}
        output['human_pose'] = human_pose
        output['human_betas'] = human_betas
        output['occ'] = occ


        if self.add_human:
            smplx_output = self.smplx_model(return_verts=True, body_pose=torch.tensor(human_pose[None, ...]),
                                            betas=torch.tensor(human_betas[None, ...]))
            vertices = smplx_output.vertices.detach().cpu().numpy().squeeze()
            joints = smplx_output.joints.detach().cpu().numpy().squeeze()
            pelvis_transform = create_mat([0, 0, 0], joints[0], rot_type='rot_vec') \
                               @ create_mat([0, np.pi, 0], np.array([0, 0, 0]), rot_type='xyz')
            vertices = trans_pcd(vertices, np.linalg.inv(pelvis_transform))
            faces = self.smplx_model.faces
            occ_human = voxelize_mesh(vertices, faces, self.grid)
            output['occ_human'] = occ_human[None, ...].to(torch.float32)

        return output, idx

    def create_smplx_model(self):
        self.smplx_model = smplx.create(MODEL_FOLDER, model_type='smplx',
                                   gender='male', ext='npz',
                                   num_betas=10,
                                   use_pca=False,
                                   create_global_orient=True,
                                   create_body_pose=True,
                                   create_betas=True,
                                   create_left_hand_pose=True,
                                   create_right_hand_pose=True,
                                   create_expression=True,
                                   create_jaw_pose=True,
                                   create_leye_pose=True,
                                   create_reye_pose=True,
                                   create_transl=True,
                                   batch_size=1)


class AhoiDatasetImage(Dataset):
    def __init__(self, data_folder, n_grid=64, len_grid=2, add_human=False):
        self.data = {}
        self.data['human_betas'] = load_data(os.path.join(data_folder, 'human_betas.npy'))
        self.data['object_location'] = load_data(os.path.join(data_folder, 'object_location.npy'))
        self.data['object_rotation'] = load_data(os.path.join(data_folder, 'object_rotation.npy'))
        self.data['object_id'] = load_data(os.path.join(data_folder, 'object_id.npy'))
        self.data['human_pose'] = load_data(os.path.join(data_folder, 'human_pose.npy'))
        self.data['img_name'] = load_data(os.path.join(data_folder, 'img_name.npy'))
        self.data['pare_bbox'] = load_data(os.path.join(data_folder, 'pare_bbox.npy'))
        self.grid = torch.from_numpy(gene_voxel_grid(N=n_grid, len=len_grid, homo=False))[None, :]
        self.n_grid = n_grid
        self.create_smplx_model()
        self.add_human = add_human

        print('Loaded Data')

    def __len__(self):
        return len(self.data['object_id'])

    def __getitem__(self, idx):
        img = Image.open(os.path.join(IMAGE_FOLDER, self.data['img_name'][idx]))

        human_pose = self.data['human_pose'][idx].astype(np.float32)
        human_betas = self.data['human_betas'][idx].astype(np.float32)
        object_rot = self.data['object_rotation'][idx].astype(np.float32)
        object_loc = self.data['object_location'][idx].astype(np.float32)
        object_id = self.data['object_id'][idx]

        matrices = create_mat_batch(object_rot, object_loc, rot_type='xyz', affine=True)
        matrices = torch.from_numpy(matrices[:, :3, :]).to(torch.float32)

        voxels = load_obj_voxel(object_id, n_grid=self.n_grid)
        voxels = torch.from_numpy(voxels).unsqueeze(1).to(torch.float32)
        grid_affine = F.affine_grid(matrices, size=[7, 1, self.n_grid, self.n_grid, self.n_grid], align_corners=False)
        voxel_affine = F.grid_sample(voxels, grid_affine, align_corners=False)
        voxel_affine = voxel_affine.view(7, self.n_grid, self.n_grid, self.n_grid)
        voxel_all = torch.any(voxel_affine, dim=0)
        occ = voxel_all[None, ...].to(torch.float32)
        occ[occ < 0] = 0.
        occ[occ > 1] = 1.

        output = {}
        bbox = self.data['pare_bbox'][idx].astype(int)
        img_orig = np.asarray(img).astype(np.float32) / 255.
        img_crop = img_orig[bbox[1] - bbox[2] // 2: bbox[1] + bbox[2] // 2, bbox[0] - bbox[2] // 2: bbox[0] + bbox[2] // 2]
        if img_crop.shape[0] < 10 or img_crop.shape[1] < 10:
            img_crop = img_orig
        try:
            img_out = train_transformer(img_crop)
        except:
            print(0)
        output['img'] = img_out
        output['human_pose'] = human_pose
        output['human_betas'] = human_betas
        output['occ'] = occ
        output['img_name'] = self.data['img_name'][idx]

        output['object_id'] = object_id

        if self.add_human:
            smplx_output = self.smplx_model(return_verts=True, body_pose=torch.tensor(human_pose[None, ...]),
                                            betas=torch.tensor(human_betas[None, ...]))
            vertices = smplx_output.vertices.detach().cpu().numpy().squeeze()
            joints = smplx_output.joints.detach().cpu().numpy().squeeze()
            pelvis_transform = create_mat([0, 0, 0], joints[0], rot_type='rot_vec') \
                               @ create_mat([0, np.pi, 0], np.array([0, 0, 0]), rot_type='xyz')
            vertices = trans_pcd(vertices, np.linalg.inv(pelvis_transform))
            faces = self.smplx_model.faces
            occ_human = voxelize_mesh(vertices, faces, self.grid)
            output['occ_human'] = occ_human[None, ...].to(torch.float32)

        return output, idx

    def create_smplx_model(self):
        self.smplx_model = smplx.create(MODEL_FOLDER, model_type='smplx',
                                   gender='male', ext='npz',
                                   num_betas=10,
                                   use_pca=False,
                                   create_global_orient=True,
                                   create_body_pose=True,
                                   create_betas=True,
                                   create_left_hand_pose=True,
                                   create_right_hand_pose=True,
                                   create_expression=True,
                                   create_jaw_pose=True,
                                   create_leye_pose=True,
                                   create_reye_pose=True,
                                   create_transl=True,
                                   batch_size=1)


class AhoiDatasetGlobal(Dataset):
    def __init__(self, data_folder, n_grid=64, len_grid=2, add_human=False):
        self.data = {}
        self.data['pare_human_betas'] = load_data(os.path.join(data_folder, 'pare_human_betas.npy'))
        self.data['human_betas'] = load_data(os.path.join(data_folder, 'human_betas.npy'))
        self.data['object_location'] = load_data(os.path.join(data_folder, 'object_location.npy'))
        self.data['object_rotation'] = load_data(os.path.join(data_folder, 'object_rotation.npy'))
        self.data['object_id'] = load_data(os.path.join(data_folder, 'object_id.npy'))
        self.data['pare_human_pose'] = load_data(os.path.join(data_folder, 'pare_human_pose.npy'))
        self.data['human_pose'] = load_data(os.path.join(data_folder, 'human_pose.npy'))
        self.data['pare_human_orient'] = load_data(os.path.join(data_folder, 'pare_human_orient.npy'))
        self.data['human_orient'] = load_data(os.path.join(data_folder, 'human_orient.npy'))
        self.data['human_transl'] = load_data(os.path.join(data_folder, 'human_transl.npy'))
        self.data['pare_cam'] = load_data(os.path.join(data_folder, 'pare_cam.npy'))

        self.data['img_name'] = load_data(os.path.join(data_folder, 'img_name.npy'))
        self.grid = torch.from_numpy(gene_voxel_grid(N=n_grid, len=len_grid, homo=False))
        self.n_grid = n_grid
        self.create_smplx_model()
        self.add_human = add_human

        print('Loaded Data')

    def __len__(self):
        return len(self.data['object_id'])

    def __getitem__(self, idx):
        human_pose = self.data['human_pose'][idx].astype(np.float32)
        pare_human_pose = self.data['pare_human_pose'][idx].astype(np.float32)
        human_betas = self.data['human_betas'][idx].astype(np.float32)
        pare_human_betas = self.data['pare_human_betas'][idx].astype(np.float32)
        human_orient = self.data['human_orient'][idx].astype(np.float32)
        pare_human_orient = self.data['pare_human_orient'][idx].astype(np.float32)
        pare_human_orient = rectify_pose(pare_human_orient)
        human_transl = self.data['human_transl'][idx].astype(np.float32)
        object_rot = self.data['object_rotation'][idx].astype(np.float32)
        object_loc = self.data['object_location'][idx].astype(np.float32)
        object_id = self.data['object_id'][idx]
        img_name = self.data['img_name'][idx]
        pare_cam = self.data['pare_cam'][idx]
        output = {}

        smplx_output = self.smplx_model(return_verts=True, body_pose=torch.tensor(pare_human_pose[None, ...]),
                                        global_orient=torch.tensor(pare_human_orient[None, ...]),
                                        transl=torch.tensor(human_transl[None, ...]),
                                        betas=torch.tensor(pare_human_betas[None, ...]))
        vertices = smplx_output.vertices.detach().cpu().numpy().squeeze()
        joints = smplx_output.joints.detach().cpu().numpy().squeeze()
        pelvis_transform_pare = create_mat(pare_human_orient, joints[0], rot_type='rot_vec') \
                           @ create_mat([0, np.pi, 0], np.array([0, 0, 0]), rot_type='xyz')
        grid = trans_pcd(self.grid, pelvis_transform_pare)
        grid_torch = torch.from_numpy(grid).unsqueeze(0)
        faces = self.smplx_model.faces
        occ_human = voxelize_mesh(vertices, faces, grid_torch)
        output['occ_pare_human'] = occ_human[None, ...].to(torch.float32)

        smplx_output = self.smplx_model(return_verts=True, body_pose=torch.tensor(human_pose[None, ...]),
                                        global_orient=torch.tensor(human_orient[None, ...]),
                                        transl=torch.tensor(human_transl[None, ...]),
                                        betas=torch.tensor(human_betas[None, ...]))
        vertices = smplx_output.vertices.detach().cpu().numpy().squeeze()
        joints = smplx_output.joints.detach().cpu().numpy().squeeze()
        pelvis_transform = create_mat(human_orient, joints[0], rot_type='rot_vec') \
                           @ create_mat([0, np.pi, 0], np.array([0, 0, 0]), rot_type='xyz')
        grid = trans_pcd(self.grid, pelvis_transform)
        grid_torch = torch.from_numpy(grid).unsqueeze(0)
        faces = self.smplx_model.faces
        occ_human = voxelize_mesh(vertices, faces, grid_torch)
        output['occ_human'] = occ_human[None, ...].to(torch.float32)


        matrices = create_mat_batch(object_rot, object_loc, rot_type='xyz', affine=True)
        matrices = torch.from_numpy(matrices[:, :3, :]).to(torch.float32)

        voxels = load_obj_voxel(object_id, n_grid=self.n_grid)
        voxels = torch.from_numpy(voxels).unsqueeze(1).to(torch.float32)
        grid_affine = F.affine_grid(matrices, size=[7, 1, self.n_grid, self.n_grid, self.n_grid], align_corners=False)
        voxel_affine = F.grid_sample(voxels, grid_affine, align_corners=False)
        voxel_affine = voxel_affine.view(7, self.n_grid, self.n_grid, self.n_grid)
        voxel_all = torch.any(voxel_affine, dim=0)
        occ = voxel_all[None, ...].to(torch.float32)
        occ[occ < 0] = 0.
        occ[occ > 1] = 1.

        output['pare_human_pose'] = pare_human_pose
        output['pare_human_betas'] = pare_human_betas
        output['pare_human_orient'] = pare_human_orient
        output['human_transl'] = human_transl
        output['occ'] = occ
        output['object_id'] = object_id
        output['pelvis_transform'] = pelvis_transform
        output['pelvis_transform_pare'] = pelvis_transform_pare
        output['img_name'] = img_name
        output['pare_cam'] = pare_cam
        output['object_id'] = object_id

        return output, idx

    def create_smplx_model(self):
        self.smplx_model = smplx.create(SMPL_MODEL_FOLDER, model_type='smplx',
                                   gender='male', ext='npz',
                                   num_betas=10,
                                   use_pca=False,
                                   create_global_orient=True,
                                   create_body_pose=True,
                                   create_betas=True,
                                   create_left_hand_pose=True,
                                   create_right_hand_pose=True,
                                   create_expression=True,
                                   create_jaw_pose=True,
                                   create_leye_pose=True,
                                   create_reye_pose=True,
                                   create_transl=True,
                                   batch_size=1)
