import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet2_utils_modified import PointNetSetAbstraction
import yaml
import models.utils as utils

class GraspEvaluatorOLDfromContinue(nn.Module):
    def __init__(self):
        super(GraspEvaluatorOLDfromContinue, self).__init__()
        self.pointnet_feature_extractor = PointnetHeader('configs/pointnet2_GRASPNET6DOF.yaml')

        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.classification = nn.Linear(1024, 2)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)

    def forward(self, pc, gripper_pc):

        # rotate and translate gripper point clouds

        # concatenate gripper_pc with pc
        pc, pc_features = self.merge_pc_and_gripper_pc(pc, gripper_pc)
        pc = pc.permute(0,2,1)
        #print(pc.shape, pc_features.shape)
        x = self.pointnet_feature_extractor(pc, pc_features)
        x = self.fc1(x)
        #print(x.shape)
        x = F.batch_norm(x,
                self.bn1.running_mean,
                self.bn1.running_var,
                self.bn1.weight,
                self.bn1.bias)
        x = torch.relu(x)
        x = self.fc2(x)
        x = F.batch_norm(x,
                self.bn2.running_mean,
                self.bn2.running_var,
                self.bn2.weight,
                self.bn2.bias)
        x = torch.relu(x)
        x = self.classification(x) # expected output Bx1
        return x[:,1] # ?
    
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

    

