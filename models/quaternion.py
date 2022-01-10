
import torch

def quaternion_conj(q):
    """
      Conjugate of quaternion q (x,y,z,w) -> (-x,-y,-z,w).
    """
    q_conj = q.clone()
    q_conj[:, :, :3] *= -1
    return q_conj

def quaternion_mult(q, r):
    """
    Multiply quaternion(s) q (x,y,z,w) with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))
    
    x = + terms[:, 3, 0] + terms[:, 2, 1] - terms[:, 1, 2] + terms[:, 0, 3]
    y = - terms[:, 2, 0] + terms[:, 3, 1] + terms[:, 0, 2] + terms[:, 1, 3]
    z = + terms[:, 1, 0] - terms[:, 0, 1] + terms[:, 3, 2] + terms[:, 2, 3]
    w = - terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] + terms[:, 3, 3] 
   

    return torch.stack((x, y, z, w), dim=1).view(original_shape)

def rot_p_by_quaterion(p, q):
    """
      Takes in points with shape of (batch_size x n x 3) and quaternions with
      shape of (batch_size x n x 4) and returns a tensor with shape of 
      (batch_size x n x 3) which is the rotation of the point with quaternion
      q. 
    """
    shape = p.shape
    q_shape = q.shape

    assert (len(shape) == 3), 'point shape = {} q shape = {}'.format(
        shape, q_shape)
    assert (shape[-1] == 3), 'point shape = {} q shape = {}'.format(
        shape, q_shape)
    assert (len(q_shape) == 3), 'point shape = {} q shape = {}'.format(
        shape, q_shape)
    assert (q_shape[-1] == 4), 'point shape = {} q shape = {}'.format(
        shape, q_shape)
    assert (q_shape[1] == shape[1]), 'point shape = {} q shape = {}'.format(
        shape, q_shape)

    q_conj = quaternion_conj(q)
    r = torch.cat([ p,
        torch.zeros(
            (shape[0], shape[1], 1), dtype=p.dtype).to(p.device)],
                  dim=-1)
    result = quaternion_mult(quaternion_mult(q, r), q_conj)
    return result[:,:,:3] 