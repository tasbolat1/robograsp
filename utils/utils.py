from re import I
import numpy as np
import trimesh.transformations as tra
import trimesh
from scipy.spatial.transform import Rotation as R

def cov_matrix(center, points):
    if points.shape[0] == 0:
        return None
    n = points.shape[0]
    diff = points - np.expand_dims(center, 0)
    cov = diff.T.dot(diff) / diff.shape[0]
    cov /= n

    eigen_values, eigen_vectors = np.linalg.eig(cov)

    order = np.argsort(-eigen_values)

    return eigen_values[order], eigen_vectors[:, order]


def choose_direction(direction, point):
    dot = np.sum(direction * point)
    if dot >= 0:
        return -direction
    return direction


def propose_grasps(pcs, radius, num_grasps=1, sampler='heuristic'):
    '''
    Heuristic and Unifrom grasp candidate generator.
    
    '''
    assert (sampler == 'uniform') or (sampler=='heuristics'), f'please indicate proper grasp candidate sampler [heuristics or uniform]. Given {sampler}.'

    if sampler=='uniform':
        grasp_translations = np.random.uniform(low=0, high=0.6, size=(num_grasps, 3))
        grasp_quaternions = R.random(num_grasps).as_quat()
        return grasp_quaternions, grasp_translations

    grasp_quaternions = []
    grasp_translations = []
    for i in range(num_grasps):
        pc = pcs[i]
        center_index = np.random.randint(pc.shape[0])
        center_point = pc[center_index, :].copy()
        d = np.sqrt(np.sum(np.square(pc - np.expand_dims(center_point, 0)),
                           -1))
        index = np.where(d < radius)[0]
        neighbors = pc[index, :]

        eigen_values, eigen_vectors = cov_matrix(center_point, neighbors)
        direction = eigen_vectors[:, 2]

        direction = choose_direction(direction, center_point)

        surface_orientation = trimesh.geometry.align_vectors([0, 0, 1],
                                                             direction)
        roll_orientation = tra.quaternion_matrix(
            tra.quaternion_about_axis(np.random.uniform(0, 2 * np.pi),
                                      [0, 0, 1]))
        gripper_transform = surface_orientation.dot(roll_orientation)
        gripper_transform[:3, 3] = center_point

        translation_transform = np.eye(4)
        translation_transform[2, 3] = -np.random.uniform(0.0669, 0.1122)

        gripper_transform = gripper_transform.dot(translation_transform)

        grasp_quaternions.append( R.from_matrix(gripper_transform[:3,:3]).as_quat() )
        grasp_translations.append( gripper_transform[:3,3] )

    return np.asarray(grasp_quaternions), np.asarray(grasp_translations)

def farthest_points(data,
                    nclusters,
                    dist_func,
                    return_center_indexes=False,
                    return_distances=False,
                    verbose=False):
    """
      Performs farthest point sampling on data points.
      Args:
        data: numpy array of the data points.
        nclusters: int, number of clusters.
        dist_dunc: distance function that is used to compare two data points.
        return_center_indexes: bool, If True, returns the indexes of the center of 
          clusters.
        return_distances: bool, If True, return distances of each point from centers.
      
      Returns clusters, [centers, distances]:
        clusters: numpy array containing the cluster index for each element in 
          data.
        centers: numpy array containing the integer index of each center.
        distances: numpy array of [npoints] that contains the closest distance of 
          each point to any of the cluster centers.
    """
    if nclusters >= data.shape[0]:
        if return_center_indexes:
            return np.arange(data.shape[0],
                             dtype=np.int32), np.arange(data.shape[0],
                                                        dtype=np.int32)

        return np.arange(data.shape[0], dtype=np.int32)

    clusters = np.ones((data.shape[0], ), dtype=np.int32) * -1
    distances = np.ones((data.shape[0], ), dtype=np.float32) * 1e7
    centers = []
    for iter in range(nclusters):
        index = np.argmax(distances)
        centers.append(index)
        shape = list(data.shape)
        for i in range(1, len(shape)):
            shape[i] = 1

        broadcasted_data = np.tile(np.expand_dims(data[index], 0), shape)
        new_distances = dist_func(broadcasted_data, data)
        distances = np.minimum(distances, new_distances)
        clusters[distances == new_distances] = iter
        if verbose:
            print('farthest points max distance : {}'.format(
                np.max(distances)))

    if return_center_indexes:
        if return_distances:
            return clusters, np.asarray(centers, dtype=np.int32), distances
        return clusters, np.asarray(centers, dtype=np.int32)

    return clusters


def distance_by_translation_grasp(p1, p2):
    """
      Gets two nx4x4 numpy arrays and computes the translation of all the
      grasps.
    """
    t1 = p1[:, :3, 3]
    t2 = p2[:, :3, 3]
    return np.sqrt(np.sum(np.square(t1 - t2), axis=-1))


def distance_by_translation_point(p1, p2):
    """
      Gets two nx3 points and computes the distance between point p1 and p2.
    """
    return np.sqrt(np.sum(np.square(p1 - p2), axis=-1))


def regularize_pc_point_count(pc, npoints, use_farthest_point=False):
    """
      If point cloud pc has less points than npoints, it oversamples.
      Otherwise, it downsample the input pc to have npoint points.
      use_farthest_point: indicates whether to use farthest point sampling
      to downsample the points. Farthest point sampling version runs slower.
    """
    if pc.shape[0] > npoints:
        if use_farthest_point:
            _, center_indexes = farthest_points(pc,
                                                npoints,
                                                distance_by_translation_point,
                                                return_center_indexes=True)
        else:
            center_indexes = np.random.choice(range(pc.shape[0]),
                                              size=npoints,
                                              replace=False)
        pc = pc[center_indexes, :]
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            index = np.random.choice(range(pc.shape[0]), size=required)
            pc = np.concatenate((pc, pc[index, :]), axis=0)
    return pc

def gripper_bd(quality=None):

    # do not change!
    gripper_line_points_main_part = np.array([
        [0.0501874312758446, -0.0000599553131906, 0.1055731475353241],
        [0.0501874312758446, -0.0000599553131906, 0.0632731392979622],
        [-0.0501874312758446, 0.0000599553131906, 0.0632731392979622],
        [-0.0501874312758446, 0.0000599553131906, 0.0726731419563293],
        [-0.0501874312758446, 0.0000599553131906, 0.1055731475353241],
    ])

    gripper_line_points_handle_part = np.array([
        [-0.0, 0.0000599553131906, 0.0632731392979622],
        [-0.0, 0.0000599553131906, -0.0032731392979622]
    ])

    if quality is not None:
        _B = quality*255.0
        _R = 255.0-_B
        _G = 0
        color = [_R, _G, _B, 255.0]
    else:
        color = None



    small_gripper_main_part = trimesh.path.Path3D(entities=[trimesh.path.entities.Line([0,1,2,3,4], color=color)],
                                                vertices = gripper_line_points_main_part)
    small_gripper_handle_part = trimesh.path.Path3D(entities=[trimesh.path.entities.Line([0, 1], color=color)],
                                                vertices = gripper_line_points_handle_part)
    small_gripper = trimesh.path.util.concatenate([small_gripper_main_part,
                                    small_gripper_handle_part])

    return small_gripper