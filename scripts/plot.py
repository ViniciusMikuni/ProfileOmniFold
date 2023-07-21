import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import tensorflow as tf
import utils
from omnifold import  Multifold,LoadJson
import tensorflow.keras.backend as K

utils.SetStyle()

parser = argparse.ArgumentParser()

parser.add_argument('--config', default='config_omnifold.json', help='Basic config file containing general options')
parser.add_argument('--plot_folder', default='../plots', help='Folder used to store plots')
parser.add_argument('--file_path', default='/global/cfs/cdirs/m3929/profile_omnifold', help='Folder containing input files')
parser.add_argument('--nevts', type=float,default=-1, help='Dataset size to use during training')
parser.add_argument('--strapn', type=int,default=0, help='Index of the bootstrap to run. 0 means no bootstrap')
parser.add_argument('--verbose', action='store_true', default=False,help='Run the scripts with more verbose output')

flags = parser.parse_args()
nevts=int(flags.nevts)
opt = LoadJson(flags.config)

data, mc_reco,mc_gen,reco_mask,gen_mask,mc_weights = utils.DataLoader(flags.file_path,opt,nevts)

for itrial in range(opt['NTRIAL']):
    mfold = Multifold(version='{}_trial{}_strapn{}'.format(opt['NAME'],itrial,flags.strapn),
                      verbose=flags.verbose)
    mfold.PrepareModel(nvars=data.shape[-1])
    mfold.LoadModel(iteration=opt['NITER']-1)
    omnifold_weights = mfold.reweight(mc_gen[gen_mask],mfold.model2)
    if itrial==0:
        weight_dict = {
            'mc gen':np.ones_like(mc_weights[gen_mask]),
            'data': omnifold_weights*mc_weights[gen_mask],
        }

        feed_dict = {
            'mc gen':mc_gen[gen_mask][:,0,0],
            'data':mc_gen[gen_mask][:,0,0], # unfolded data in omnifold is Sim + data weights
        }

        fig,ax = utils.HistRoutine(feed_dict,plot_ratio=True,
                                   weights = weight_dict,
                                   binning=np.linspace(0.1,1.0,40),
                                   xlabel='Leading subjet pT [TeV]',logy=True,
                                   ylabel='Normalized events',
                                   reference_name='data')
        fig.savefig('{}/{}_{}.pdf'.format(flags.plot_folder,"Unfolded_Hist",opt['NAME']))
        

