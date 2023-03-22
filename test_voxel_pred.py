import torch.nn as nn
from torch.utils.data import DataLoader
from dataloaders import AhoiDatasetImage
from models_voxel_pred import VoxelPredNet
from tqdm import tqdm
from visualize import *



if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    smplx_model = smplx.create(SMPL_MODEL_FOLDER, model_type='smplx',
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



    # Hyper-parameters
    input_size = 63
    hidden_size = 500
    output_size = 22
    num_epochs = 100
    batch_size = 8
    learning_rate = 0.00005
    n_grid = 64

    grid = gene_voxel_grid(N=n_grid, len=2, homo=False)

    test_dataset = AhoiDatasetImage(data_folder=DATA_FOLDER, add_human=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

    model = VoxelPredNet(hidden_size=16).to(device)
    MODEL_PATH = os.path.join(MODEL_FOLDER, 'model_voxel_pred.pth')
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500, 2)

    # Train the model
    total_step = len(test_loader)

    step_id = 0
    chamfer_list = []
    iou_list = []
    pene_list = []
    cont_list = []
    with torch.no_grad():

        for i, (output, idx) in enumerate(tqdm(test_loader)):
            model.train()
            step_id += 1
            img = output['img'].to(device)
            occ_human = output['occ_human'].to(device)
            occ = output['occ'].to(device)
            img_name = output['img_name'][0]

            occ_pred = model(img, occ_human)

            human_pose = output['human_pose']
            human_orient = output['human_orient']
            human_transl = output['human_transl']

            human_pose = torch.from_numpy(human_pose).view(1, -1)
            human_orient = torch.from_numpy(human_orient).view(1, -1)
            human_transl = torch.from_numpy(human_transl).view(1, -1)

            smplx_output = smplx_model(return_verts=True, body_pose=human_pose, global_orient=human_orient, transl=human_transl)

            pcd_obj = voxel_to_pcd(grid, occ_pred)
            pcd_human = voxel_to_pcd(grid, occ_human)
            vertices = smplx_output.vertices.detach().cpu().numpy().squeeze()
            joints = smplx_output.joints.detach().cpu().numpy().squeeze()
            pelvis_transform = create_mat(human_orient, joints[0], rot_type='rot_vec') \
                               @ create_mat([0, np.pi, 0], np.array([0, 0, 0]), rot_type='xyz')
            faces = smplx_model.faces
            mesh_human = trimesh.Trimesh(vertices.squeeze(), faces=faces.squeeze())
            mesh_human.visual.vertex_colors = [95, 158, 160]
            occ_pred[occ_pred > 0.5] = 1
            occ_pred[occ_pred <= 0.5] = 0
            mesh_obj = trimesh.voxel.ops.matrix_to_marching_cubes(occ_pred.squeeze().detach().cpu().numpy(), pitch=2 / n_grid)
            mesh_obj.apply_transform(create_mat([0, 0, 0], [-1., -1., -1.], rot_type='xyz'))
            mesh_obj.visual.vertex_colors = [216, 191, 216]
            mesh_obj.apply_transform(pelvis_transform)

            visualize_result(extra_meshes=[mesh_obj, mesh_human], show=True)





