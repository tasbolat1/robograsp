from tqdm.std import tqdm
import torch
import numpy as np
from models.models import GraspSamplerDecoder, GraspEvaluator
from utils import utils
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm
import time

class GRASPNET(object):
    def __init__(self, args):
        super(GRASPNET, self).__init__()
        self.args = args
        if args['device'] == -1:
            self.device = 'cpu'
        else:
            self.device = torch.device(f'cuda:{args["device"]}')
        self.batch_size = self.args['batch_size']

    def sample_and_refine_grasps(self, pc):
        # expected pc is [N, 3] relative to camera in numpy

        # prepare point cloud
        pc, pc_mean = self.prepare_pointclouds(pc)

        pc, pc_mean, quaternions, translations = self.sample_grasps(pc, pc_mean)
        
        # load evaluator
        self.load_evaluator()

        # convert to eulers
        eulers = self.quaternions2eulers(quaternions)
        eulers, translations, final_success = self.refine_grasps(pc, pc_mean, eulers, translations)
        eulers = eulers.cpu().numpy()
        translations = translations.cpu().numpy()

        quaternions = self.eulers2quaternions(eulers)

        return quaternions, translations, final_success

    def refine_grasps(self, pc, pc_mean, eulers, translations):
        # sample grasps
        

        split_size = int(self.args['n']/(self.batch_size+1))+1
        split_indcs = np.array_split(np.arange(self.args['n']), split_size)
        #print(split_indcs)
        # refine by batch

        print(f'Start to refine using DGflow ... .')
        start_time = time.time()
        inital_success = np.zeros(translations.shape[0])
        final_success = np.zeros(translations.shape[0])
        for indcs in tqdm( split_indcs ):
            _pc, _eulers, _translations = pc[indcs], eulers[indcs], translations[indcs]

            # save initial values

            for t in tqdm( range(1, self.args['max_iterations']+1), leave=False):
                # get gradient
                eulers_v, translations_v, success = self._velocity(_eulers,_translations, Nq=self.args['Nq'], Np=self.args['Np'], pc=_pc)
                
                # update
                _eulers = _eulers.data + self.args['eta_eulers'] * eulers_v + \
                                np.sqrt(2*self.args['eta_eulers']) * self.args['noise_factor'] * torch.randn_like(_eulers)
                _translations = _translations.data + self.args['eta_trans'] * translations_v +\
                                np.sqrt(2*self.args['eta_trans']) * self.args['noise_factor'] * torch.randn_like(_translations)

                if t == 1:
                    inital_success[indcs] = success.detach().squeeze(1).cpu().numpy()

            eulers[indcs] = _eulers.detach()
            translations[indcs] = _translations.detach() + pc_mean
            final_success[indcs] = success.detach().squeeze(1).cpu().numpy()
        end_time = time.time()

        print(f'Refinemet finished in {end_time-start_time} seconds for {self.args["n"]} samples. ')

        # show stats
        print(f'Average initial success: {np.mean(inital_success)}')
        print(f'Average final success: {np.mean(final_success)}')

        return eulers, translations, final_success


    def sample_grasps(self , pc, pc_mean):

        print(f'Sampling grasps with {self.args["sampler"]}')
        
        net = None
        if ('VAE' in self.args['sampler']) or ('GAN' in self.args['sampler']):
            net = GraspSamplerDecoder(config_path='configs/pointnet2_GRASPNET6DOF_sampler.yaml').to(self.device)
            net.load_state_dict(torch.load(f'models/pretrained_auxilariy_models/{self.args["sampler"]}_DECODER_latest.pt'))
            net.eval()

            
            split_size = int(self.args['n']/(self.batch_size+1))+1
            pcs_indcs = np.array_split(np.arange(self.args['n']), split_size)
            quaternions, translations = [], []
            for indcs in pcs_indcs:
                with torch.no_grad():
                    _pcs = pc[indcs]
                    quats, trans = net(_pcs)
                    quaternions.append(quats)
                    translations.append(trans)

            quaternions = torch.cat(quaternions)
            translations = torch.cat(translations)
            print(pc.shape, pc_mean.shape, quaternions.shape, translations.shape)

            # ADHOC: error if tranlsations are too far:
            translations_norm = torch.norm(translations, dim=1, keepdim=True)
            mask = translations_norm >= 1 # more than one meter ?
            mask.squeeze_()
            translations[mask,:] = translations[mask,:].div(translations_norm[mask])

            # these translations are shifted
            # pc: [B,1024, 3], pc_mean [B,3], quaternions [B,4], translations [B,3]
            del net
            torch.cuda.empty_cache()

            print(f'Done with sampling')
            return pc, pc_mean, quaternions, translations

        elif (self.args['sampler'] == 'heuristics') or (self.args['sampler'] == 'uniform'):
            quaternions, translations = utils.propose_grasps(pc, radius=0.01, num_grasps=self.args['n'], sampler=self.args['sampler'])
            quaternions = torch.FloatTensor(quaternions).to(self.device)
            translations = torch.FloatTensor(translations).to(self.device)
            pc = torch.FloatTensor(pc).to(self.device)
            pc_mean = torch.FloatTensor(pc_mean).to(self.device)
            # print(pc.shape, pc_mean.shape, quaternions.shape, translations.shape)
            print(f'Done with sampling')
            return pc, pc_mean, quaternions, translations
        else:
            raise NotImplementedError


    def prepare_pointclouds(self, pc):
        # expected pc is [N, 3] relative to camera in numpy
        # return: pcs [n, 1024, 3], pcs_mean [n, 1, 3]

        pc = utils.regularize_pc_point_count(pc, 1024) # [1024, 3]
        pc = np.expand_dims(pc, axis=0)
        pcs = np.repeat(pc, repeats=self.args['n'], axis=0)
        pcs_mean = pcs.mean(axis = 1)

        pcs = torch.FloatTensor(pcs).to(self.device)
        pcs_mean = torch.FloatTensor(pcs_mean).to(self.device)
        print(pcs.shape, pcs_mean.shape)
        return pcs, pcs_mean

    def load_evaluator(self):
        self.evaluator = GraspEvaluator(config_path='configs/pointnet2_GRASPNET6DOF_evaluator.yaml',
                           control_point_path='configs/panda.npy').to(self.device)
        model_path = self.args['model_path']
        self.evaluator.load_state_dict(torch.load(model_path))
        self.evaluator.eval()
        
    def _velocity(self, eulers, translations, pc, Nq=1, Np=1, f=None):

        # eulers [B,3]
        # translations [B,3]
        # pc [B,1024,3]

        eulers_t = eulers.clone()
        translations_t = translations.clone()
        eulers_t.requires_grad_(True)
        translations_t.requires_grad_(True)
        if eulers_t.grad is not None:
            eulers_t.grad.zero_()
        if translations_t.grad is not None:
            translations_t.grad.zero_()
            
        pc.required_grad = False
        d_score = self.evaluator.forward_with_eulers(eulers_t, translations_t, pc)
        success = torch.sigmoid(d_score)

        Nq = torch.FloatTensor([Nq]).to(self.device)
        Np = torch.FloatTensor([Np]).to(self.device)
        bias_term = torch.log(Nq) - torch.log(Np)
        d_score -= bias_term

        if self.args['f'] == 'KL':
            s_eulers = torch.ones_like(d_score.detach())
            s_translations = torch.ones_like(d_score.detach())

        elif self.args['f'] == 'logD':
            s_eulers = 1 / (1 + d_score.detach().exp())
            s_translations = 1 / (1 + d_score.detach().exp())

        elif self.args['f'] == 'JS':
            s_eulers = 1 / (1 + 1 / d_score.detach().exp())
            s_translations = 1 / (1 + 1 / d_score.detach().exp())
        else:
            raise ValueError()

        s_eulers.expand_as(eulers_t)
        s_translations.expand_as(translations_t)
        d_score.backward(torch.ones_like(d_score).to(self.device))
        eulers_grad = eulers_t.grad
        trans_grad = translations_t.grad
        return s_eulers * eulers_grad.data, \
               s_translations.data * trans_grad.data, \
               success

    def quaternions2eulers(self, quaternions):
        r = R.from_quat(quaternions.cpu().numpy())
        eulers = torch.FloatTensor(r.as_euler(seq='XYZ')).to(self.device)
        return eulers

    def eulers2quaternions(self, eulers):
        r = R.from_euler(angles = eulers, seq='XYZ')
        quaternions = r.as_quat()
        return quaternions
