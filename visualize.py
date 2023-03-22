import smplx
from trimesh import creation
from ahoi_utils import *
import kaolin


def visualize_result(human_pose=None, human_betas=None, human_transl=None, human_orient=None, object_rotation=None, object_location=None, joint_prox=None, object_id=None, rigid=False, show_axis=False, voxel_n=0, extra_meshes=[], save_name='', return_img=False, show=True):

    mg = gene_voxel_grid(N=voxel_n) if voxel_n else None
    origin_axis = creation.axis()
    meshes = [origin_axis] if show_axis else []

    if human_pose is not None:
        model = smplx.create(SMPL_MODEL_FOLDER, model_type='smplx',
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
                             batch_size=1,
                             )



        torch_param = {}
        body_pose = human_pose.reshape((63,))[None, :].astype(np.float32)
        torch_param['body_pose'] = torch.tensor(body_pose)
        if human_betas is not None:
            betas = human_betas.reshape((10,))[None, :].astype(np.float32)
            torch_param['betas'] = torch.tensor(betas)
        if human_transl is not None:
            transl = human_transl.reshape((3,))[None, :].astype(np.float32)
            torch_param['transl'] = torch.tensor(transl).detach()
        if human_orient is not None:
            global_orient = human_orient.reshape((3,))[None, :].astype(np.float32)
            torch_param['global_orient'] = torch.tensor(global_orient)
        else:
            global_orient = np.zeros((1, 3))

        output = model(return_verts=True, **torch_param)
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        joints = output.joints.detach().cpu().numpy().squeeze()



        human_mesh = trimesh.Trimesh(vertices=vertices, faces=model.faces)
        # human_mesh.visual.vertex_colors = [240, 255, 255]
        pelvis_axis = creation.axis()
        pelvis_transform = create_mat(global_orient[0], joints[0], rot_type='rot_vec') \
                           @ create_mat([0, np.pi, 0], np.array([0, 0, 0]), rot_type='xyz')
        pelvis_axis.apply_transform(pelvis_transform)

        if joint_prox is not None:
            for ind, joint in enumerate(joints[:22]):
                sphere = creation.uv_sphere(radius=joint_prox[ind] / 10.)
                trans = create_mat([0, 0, 0], joint, rot_type='xyz')
                sphere.apply_transform(trans)
                sphere.visual.vertex_colors = [255, 10, 10]
                meshes.append(sphere)
        else:
            meshes += [human_mesh, pelvis_axis] if show_axis else [human_mesh]

    else:
        joints = None
        pelvis_transform = create_mat(human_orient if human_orient is not None else [0, 0, 0], [0, 0, 0], rot_type='xyz')

    # meshes += [human_mesh, pelvis_axis]

    if object_rotation is not None and object_location is not None:
        obj_meta = load_obj_meta(array=True)
        object_ind_in_meta = np.where(obj_meta['object_ids'] == int(object_id))[0][0]
        if voxel_n:
            grid = mg @ pelvis_transform.T
            grid = grid[:, :3]
            grid = torch.from_numpy(grid[None, ...])

        occ_all = torch.zeros(voxel_n ** 3)
        for pid in range(len(PART_LIST)):
            face_len = obj_meta['face_len'][object_ind_in_meta][pid]
            if not face_len:
                continue
            if not rigid:
                rot = object_rotation[pid]
                loc = object_location[pid]
                M = create_mat(rot, loc, rot_type='xyz')
                M = pelvis_transform @ M

            else:
                rot = object_rotation.reshape(3,)
                loc = object_location.reshape(3,)
                init_shift = obj_meta['init_shift'][object_ind_in_meta][pid]
                M = create_mat(rot, loc, rot_type='xyz') \
                    @ create_mat([0, 0, 0], init_shift, rot_type='xyz')
                M = pelvis_transform @ M

            if voxel_n:
                vertices, faces = load_mesh(object_meta=obj_meta, object_id=object_id, part_id=pid, trans_M=M, to_trimesh=False)
                vertices = torch.from_numpy(vertices[None, ...])
                faces = torch.from_numpy(faces)
                occ = kaolin.ops.mesh.check_sign(vertices, faces, grid)
                occ_all = torch.logical_or(occ.squeeze(), occ_all)
            else:
                part_mesh = load_mesh(object_meta=obj_meta, object_id=object_id, part_id=pid, trans_M=M, to_trimesh=True)
                meshes.append(part_mesh)
        if voxel_n:
            pcd = trimesh.points.PointCloud(vertices=grid[0][occ_all])
            meshes.append(pcd)

    meshes += [m.apply_transform(pelvis_transform) for m in extra_meshes]
    # meshes += [m for m in extra_meshes]

    if show:
        scene = trimesh.Scene(meshes)
        scene.set_camera((0, 1.5, 0), 3, joints[0] if joints is not None else [0, 0, 0])
        scene.show()
    if save_name or return_img:
        scene = trimesh.Scene(meshes)
        scene.set_camera((0, 1.0, 0), 3, [0, 0, 0])
        png = scene.save_image(resolution=[640, 480], visible=True)
        if return_img:
            return png
        with open(save_name, 'wb') as f:
            f.write(png)

    return meshes


def voxelize_numpy(object_meta, object_id, grid, object_rotation, object_location, rigid=False):
    object_ind_in_meta = np.where(object_meta['object_ids'] == int(object_id))[0][0]
    occ_all = torch.zeros(grid.shape[1])
    for pid in range(len(PART_LIST)):
        face_len = object_meta['face_len'][object_ind_in_meta][pid]
        if not face_len:
            continue
        if not rigid:
            rot = object_rotation[pid]
            loc = object_location[pid]
            M = create_mat(rot, loc, rot_type='xyz')
            
        else:
            rot = object_rotation.reshape(3, )
            loc = object_location.reshape(3, )
            init_shift = object_meta['init_shift'][object_ind_in_meta][pid]
            M = create_mat(rot, loc, rot_type='xyz') \
                @ create_mat([0, 0, 0], init_shift, rot_type='xyz')
        vertices, faces = load_mesh(object_meta=object_meta, object_id=object_id, part_id=pid, trans_M=M,
                                    to_trimesh=False)
        vertices = torch.from_numpy(vertices[None, ...])
        faces = torch.from_numpy(faces)
        occ = kaolin.ops.mesh.check_sign(vertices, faces, grid)
        occ_all = torch.logical_or(occ.squeeze(), occ_all)

    grid_len = int(grid.shape[1] ** (1/3) + 0.00001)
    occ_all = occ_all.view(grid_len, grid_len, grid_len)

    return occ_all


def voxelize_mesh(vertices, faces, grid):


    if isinstance(vertices, np.ndarray):
        vertices = torch.from_numpy(vertices[None, ...])
    else:
        vertices = vertices[None, ...]
    if isinstance(faces, np.ndarray):
        faces = torch.from_numpy(faces.astype(int))
    occ = kaolin.ops.mesh.check_sign(vertices, faces, grid)
    grid_len = int(grid.shape[1] ** (1/3) + 0.00001)
    occ_all = occ.view(grid_len, grid_len, grid_len)

    return occ_all



