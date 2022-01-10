import numpy as np
import yaml
from graspnet import GRASPNET


args = yaml.safe_load(open('configs/robograsp.yaml', 'r'))
pc = np.random.rand(2000, 3)

graspnet = GRASPNET(args)

graspnet.sample_and_refine_grasps(pc)