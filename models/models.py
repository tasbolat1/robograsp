import torch
import torch.nn as nn

from .pointnet2_utils import PointNetSetAbstraction
import yaml
import models.utils as utils
import torch.nn.functional as F


class GraspEvaluator(nn.Module):
    def __init__(self, config_path='configs/pointnet2_GRASPNET6DOF_evaluator.yaml', control_point_path='configs/panda.npy'):
        super(GraspEvaluator, self).__init__()
        self.pointnet_feature_extractor = PointnetHeader(config_path)

        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.classification = nn.Linear(1024, 1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)

        self.control_point_path = control_point_path


    def forward(self, quat, trans, pc):
        '''
        Input:
        either: quat: [B,4]
        trans: [B,3]
        pc: [B,1024,3]
        '''
        # rotate and translate gripper point clouds
        gripper_pc = utils.transform_gripper_pc_old(quat, trans, config_path=self.control_point_path)            
            
        return self.evaluate_grasp(pc, gripper_pc)

    def forward_with_eulers(self, eulers, trans, pc):
        '''
        Input:
        either: euler: [B,3]
        trans: [B,3]
        pc: [B,1024,3]
        '''
        gripper_pc = utils.control_points_from_rot_and_trans(eulers, trans, config_path=self.control_point_path)
        return self.evaluate_grasp(pc, gripper_pc)

        
    def evaluate_grasp(self, pc, gripper_pc):
        # concatenate gripper_pc with pc
        pc, pc_features = self.merge_pc_and_gripper_pc(pc, gripper_pc)
        pc = pc.permute(0,2,1)
        #print(pc.shape, pc_features.shape)
        x = self.pointnet_feature_extractor(pc, pc_features)
        x = self.fc1(x)
        #print(x.shape)
        x = torch.relu(self.bn1(x))
        x = self.fc2(x)
        x = torch.relu(self.bn2(x))
        x = self.classification(x) # expected output Bx1
        return x
    
    def merge_pc_and_gripper_pc(self, pc, gripper_pc):
        """
        Merges the object point cloud and gripper point cloud and
        adds a binary auxiliary feature that indicates whether each point
        belongs to the object or to the gripper.
        """
        pc_shape = pc.shape
        gripper_shape = gripper_pc.shape
        assert (len(pc_shape) == 3)
        assert (len(gripper_shape) == 3)
        assert (pc_shape[0] == gripper_shape[0])

        npoints = pc_shape[1]
        batch_size = pc_shape[0]

        l0_xyz = torch.cat((pc, gripper_pc), 1)
        labels = [
            torch.ones(pc.shape[1], 1, dtype=torch.float32),
            torch.zeros(gripper_pc.shape[1], 1, dtype=torch.float32)
        ]
        labels = torch.cat(labels, 0)
        labels.unsqueeze_(0)
        labels = labels.repeat(batch_size, 1, 1)

        l0_points = torch.cat([l0_xyz, labels.to(pc.device)],
                              -1).transpose(-1, 1)
        return l0_xyz, l0_points


class GraspSamplerDecoder(nn.Module):
    def __init__(self, config_path='configs/pointnet2_GRASPNET6DOF_sampler.yaml'):
        '''
        Architecture and weights are taken from 6dof graspnet paper
        '''
        super(GraspSamplerDecoder, self).__init__()
        self.pointnet_feature_extractor = PointnetHeader(config_path)

        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.q = nn.Linear(1024, 4)
        self.t = nn.Linear(1024, 3)
        
    def forward(self, pc, z=None):

        # rotate and translate gripper point clouds
        # concatenate gripper_pc with pc
        if z is None:
            z = torch.randn(pc.shape[0], 2).to(pc.device)
        xyz_features = self.concatenate_z_with_pc(pc, z)
        pc = pc.permute(0,2,1)
        xyz_features = xyz_features.permute(0,2,1)
        #print(pc.shape, xyz_features.shape)
        x = self.pointnet_feature_extractor(pc, xyz_features)
        x = self.fc1(x)
        x = torch.relu(self.bn1(x))
        x = self.fc2(x)
        x = torch.relu(self.bn2(x))
        #q = self.q(x)
        q = F.normalize(self.q(x), p=2, dim=-1)
        t = self.t(x)
        return q, t
    
    def concatenate_z_with_pc(self, pc, z):
        z.unsqueeze_(1)
        z = z.expand(-1, pc.shape[1], -1)
        return torch.cat((pc, z), -1)

class PointnetHeader(nn.Module):
    def __init__(self, path_to_cfg):
        super(PointnetHeader, self).__init__()

        # load pointnet header configs
        pointnet_params = yaml.safe_load(open(path_to_cfg, 'r'))
        
        self.pointnet_modules = nn.ModuleList()
        for _, params in pointnet_params.items():
            # we use positions as features also
            in_channel = params['in_channel'] + 3
            sa_module = PointNetSetAbstraction(npoint=params['npoint'],
                                            radius=params['radius'],
                                            nsample=params['nsample'],
                                            in_channel=in_channel,
                                            mlp=params['mlp'],
                                            group_all=params['group_all'])
            self.pointnet_modules.append( sa_module )

    def forward(self, points, point_features):
        for pointnet_layer in self.pointnet_modules:
            points, point_features = pointnet_layer(points, point_features)
        return point_features.squeeze(-1)

    

