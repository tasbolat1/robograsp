import torch
import numpy as np
from pathlib import Path
import models.quaternion as quaternion

def save_model(model, path, epoch=None, reason=None):
    
    if reason is None:
        print(f'saving model at {epoch} epoch ...')
        save_path = Path(path)/f'{epoch}.pt'
    else:
        print(f'saving model at {epoch} epoch at reason {reason}...')
        save_path = Path(path)/f'{reason}.pt'

    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)


def get_gripper_pc(use_torch=True, path='configs/gripper_pc.npy', full=False):
    """
      Outputs a tensor of shape (batch_size x 28 x 3).
      use_tf: switches between outputing a tensor and outputing a numpy array.
    """
    if full:
        path = 'configs/full_gripper_pc.npy'
        
    control_points = np.load(path)
    control_points = np.asarray(control_points, dtype=np.float32)

    if use_torch:
        return torch.FloatTensor(control_points)

    return control_points


def translation_distance(trans1, trans2):
    '''
    Input Bx3
    Output B
    '''
    return torch.sqrt( torch.sum( (trans1 - trans2)**2, dim=1) )

def quaternion_distance(quat1, quat2):
    '''
    Input Bx4
    Output B
    '''
    return 2*torch.arccos( torch.abs( torch.sum(quat1 * quat2, dim=1) ) )

def grasp_distance(quat1, trans1, quat2, trans2, use_3d_points=False, norm='l2', quat_weight=1, trans_weight=1):
    '''
    Calculated distance between two grasps
    Input:
        quat1, quat2: [Bx4] vectors
        trans1, trans2: [Bx3] vectors
        use_3d_points: Bool, set it True if distance is calculated by gripper points.
        norm: 'l1' or 'l2', only true if use_3d_points set True.
        quat_weight: weight for quaternion distance
        trans weight: weight for translation distance
    Returns:
        distance: [B] distance between grasps
    '''
    if use_3d_points:
        gripper_pc1 = transform_gripper_pc(quat1, trans1)
        gripper_pc2 = transform_gripper_pc(quat2, trans2)

        if norm == 'l2':
            return torch.sqrt( torch.sum( (gripper_pc1 - gripper_pc2)**2, dim=(1,2)) )
        elif norm == 'l1':
            return torch.sum( torch.abs(gripper_pc1 - gripper_pc2), dim=(1,2))
        else:
            raise ValueError()

    else:
        trans_distance = torch.mean( translation_distance(trans1, trans2) )
        quat_distance = torch.mean( quaternion_distance(quat1, quat2) )

        return trans_weight*trans_distance + quat_weight*quat_distance


def transform_gripper_pc_old(quat, trans, config_path = 'configs/panda.npy'):
    # q: (x,y,z, w)
    # t: (x,y,z)
    
    # upload gripper_pc
    control_points = np.load(config_path)[:, :3]
    control_points = [[0, 0, 0], [0, 0, 0], control_points[0, :],
                      control_points[1, :], control_points[-2, :],
                      control_points[-1, :]]
    control_points = np.asarray(control_points, dtype=np.float32)
    control_points = np.tile(np.expand_dims(control_points, 0),
                             [quat.shape[0], 1, 1])

    gripper_pc = torch.tensor(control_points).to(quat.device)

    # prepare q and t 
    quat = quat.unsqueeze(1).repeat([1, gripper_pc.shape[1], 1])
    trans = trans.unsqueeze(1).repeat([1, gripper_pc.shape[1], 1])

    # rotate and add
    gripper_pc = quaternion.rot_p_by_quaterion(gripper_pc, quat)
    gripper_pc +=trans

    return gripper_pc

def transform_gripper_pc(quat, trans, full=False):
    # q: (x,y,z, w)
    # t: (x,y,z)
    
    # upload gripper_pc
    gripper_pc = get_gripper_pc(full=full).to(quat.device)
    gripper_pc = gripper_pc.unsqueeze(0).to(quat.device)
    gripper_pc = gripper_pc.repeat([quat.shape[0], 1, 1])

    # prepare q and t 
    quat = quat.unsqueeze(1).repeat([1, gripper_pc.shape[1], 1])
    trans = trans.unsqueeze(1).repeat([1, gripper_pc.shape[1], 1])

    # rotate and add
    gripper_pc = quaternion.rot_p_by_quaterion(gripper_pc, quat)
    gripper_pc +=trans

    return gripper_pc

def concatenate_pcs(object_pc, gripper_pc):
    """
    input:
        object_pc: [B,N1,3], N1 per object pointcloud
        gripper_pc: [B,N2,3], N2 per gripper pointcloud
    Merges the object point cloud and gripper point cloud and
    adds a binary auxiliary feature that indicates whether each point
    belongs to the object or to the gripper.
    return:
        points: [B,3,N1+N2]
        point_features: [B,4,N1+N2]
    """
    # pc_shape = object_pc.shape
    # gripper_shape = gripper_pc.shape
    # assert (len(pc_shape) == 3)
    # assert (len(gripper_shape) == 3)
    # assert (pc_shape[0] == gripper_shape[0])

    points = torch.cat((object_pc, gripper_pc), 1)
    labels = [
        torch.ones(object_pc.shape[1], 1, dtype=torch.float32),
        torch.zeros(gripper_pc.shape[1], 1, dtype=torch.float32)
    ]
    labels = torch.cat(labels, 0).unsqueeze(0)
    labels = labels.repeat(object_pc.shape[0], 1, 1)

    point_features = torch.cat([points, labels.to(object_pc.device)],
                            -1).transpose(-1, 1)
    return points.permute(0, 2, 1), point_features

def convert_Avec_to_A(A_vec):
    """ Convert BxM tensor to BxNxN symmetric matrices """
    """ M = N*(N+1)/2"""
    if A_vec.dim() < 2:
        A_vec = A_vec.unsqueeze(dim=0)
    
    if A_vec.shape[1] == 10:
        A_dim = 4
    elif A_vec.shape[1] == 55:
        A_dim = 10
    else:
        raise ValueError("Arbitrary A_vec not yet implemented")

    idx = torch.triu_indices(A_dim,A_dim)
    A = A_vec.new_zeros((A_vec.shape[0],A_dim,A_dim))   
    A[:, idx[0], idx[1]] = A_vec
    A[:, idx[1], idx[0]] = A_vec
    return A.squeeze()


def control_points_from_rot_and_trans(grasp_eulers,
                                      grasp_translations, config_path='configs/panda.npy'):
    rot = tc_rotation_matrix(grasp_eulers[:, 0],
                             grasp_eulers[:, 1],
                             grasp_eulers[:, 2],
                             batched=True)
    grasp_pc = get_control_point_tensor(grasp_eulers.shape[0], config_path=config_path).to(grasp_eulers.device)
    grasp_pc = torch.matmul(grasp_pc, rot.permute(0, 2, 1))
    grasp_pc += grasp_translations.unsqueeze(1).expand(-1, grasp_pc.shape[1],
                                                       -1)
    return grasp_pc

def tc_rotation_matrix(az, el, th, batched=False):
    if batched:

        cx = torch.cos(torch.reshape(az, [-1, 1]))
        cy = torch.cos(torch.reshape(el, [-1, 1]))
        cz = torch.cos(torch.reshape(th, [-1, 1]))
        sx = torch.sin(torch.reshape(az, [-1, 1]))
        sy = torch.sin(torch.reshape(el, [-1, 1]))
        sz = torch.sin(torch.reshape(th, [-1, 1]))

        ones = torch.ones_like(cx)
        zeros = torch.zeros_like(cx)

        rx = torch.cat([ones, zeros, zeros, zeros, cx, -sx, zeros, sx, cx],
                       dim=-1)
        ry = torch.cat([cy, zeros, sy, zeros, ones, zeros, -sy, zeros, cy],
                       dim=-1)
        rz = torch.cat([cz, -sz, zeros, sz, cz, zeros, zeros, zeros, ones],
                       dim=-1)

        rx = torch.reshape(rx, [-1, 3, 3])
        ry = torch.reshape(ry, [-1, 3, 3])
        rz = torch.reshape(rz, [-1, 3, 3])

        return torch.matmul(rz, torch.matmul(ry, rx))
    else:
        cx = torch.cos(az)
        cy = torch.cos(el)
        cz = torch.cos(th)
        sx = torch.sin(az)
        sy = torch.sin(el)
        sz = torch.sin(th)

        rx = torch.stack([[1., 0., 0.], [0, cx, -sx], [0, sx, cx]], dim=0)
        ry = torch.stack([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dim=0)
        rz = torch.stack([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dim=0)

        return torch.matmul(rz, torch.matmul(ry, rx))

def get_control_point_tensor(batch_size, use_torch=True, config_path = 'configs/panda.npy'):
    """
      Outputs a tensor of shape (batch_size x 6 x 3).
      use_tf: switches between outputing a tensor and outputing a numpy array.
    """
    control_points = np.load(config_path)[:, :3]
    control_points = [[0, 0, 0], [0, 0, 0], control_points[0, :],
                      control_points[1, :], control_points[-2, :],
                      control_points[-1, :]]
    control_points = np.asarray(control_points, dtype=np.float32)
    control_points = np.tile(np.expand_dims(control_points, 0),
                             [batch_size, 1, 1])

    if use_torch:
        return torch.tensor(control_points)

    return control_points