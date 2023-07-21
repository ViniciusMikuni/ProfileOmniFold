import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import argparse
import os
import horovod.tensorflow.keras as hvd
import tensorflow as tf
import uproot3 as uproot

line_style = {
    'data reco':'-',
    'mc gen':'-',
    'mc reco':'-',
    'data':'dotted',
}


colors = {
    'data':'black',
    'data reco':'black',
    'mc gen':'#7570b3',
    'mc reco':'#d95f02',
}


            
def FormatFig(xlabel,ylabel,ax0):
    #Limit number of digits in ticks
    # y_loc, _ = plt.yticks()
    # y_update = ['%.1f' % y for y in y_loc]
    # plt.yticks(y_loc, y_update) 
    ax0.set_xlabel(xlabel,fontsize=20)
    ax0.set_ylabel(ylabel)

def SetStyle():
    from matplotlib import rc
    rc('text', usetex=True)

    import matplotlib as mpl
    rc('font', family='serif')
    rc('font', size=22)
    rc('xtick', labelsize=15)
    rc('ytick', labelsize=15)
    rc('legend', fontsize=15)

    # #
    mpl.rcParams.update({'font.size': 19})
    #mpl.rcParams.update({'legend.fontsize': 18})
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams.update({'xtick.labelsize': 18}) 
    mpl.rcParams.update({'ytick.labelsize': 18}) 
    mpl.rcParams.update({'axes.labelsize': 18}) 
    mpl.rcParams.update({'legend.frameon': False}) 
    mpl.rcParams.update({'lines.linewidth': 2})
    
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.set_style(hep.style.CMS)
    hep.style.use("CMS") 

def SetGrid(ratio=True):
    fig = plt.figure(figsize=(9, 9))
    if ratio:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1]) 
        gs.update(wspace=0.025, hspace=0.1)
    else:
        gs = gridspec.GridSpec(1, 1)
    return fig,gs

def convert_to_polar(vec):
    new_vec = np.zeros_like(vec)
    mask = vec!=0
    
    new_vec[:,:,0] = np.sqrt(vec[:,:,0]**2 + vec[:,:,1]**2)/1000. #Normalize to make the training easier
    new_vec[:,:,1] = np.ma.arcsinh(np.ma.divide(vec[:,:,2],new_vec[:,:,0]).filled(0)).filled(0)
    new_vec[:,:,2] = np.ma.arctan2(vec[:,:,1],vec[:,:,0]).filled(0)
    new_vec[:,:,3] = vec[:,:,3]/1000.
    new_vec = new_vec*mask
    return new_vec

def DataLoader(base_path,config,nevts=-1):
    hvd.init()

    mc_gen = []
    mc_reco = []
    reco_mask = []
    gen_mask = []
    mc_weights = []

    data_reco = []
    data_weights = []
    
    #Those are the features used by OmniFold. All of them and any function of them are also unfolded at the end of the method!
    feature_list = [
        'px_sub1','py_sub1','pz_sub1','E_sub1',
        'px_sub2','py_sub2','pz_sub2','E_sub2',
        'px_sub3','py_sub3','pz_sub3','E_sub3',        
    ]    
    #Read the MC files
    for sample in config['FILE_MC']:
        file_path = os.path.join(base_path,sample)
        tmp_file = uproot.open(file_path)['AnalysisTree']
        # print(tmp_file.keys())
        mcr = [tmp_file[feat+'_rec'].array() for feat in feature_list]
        mcg = [tmp_file[feat+'_gen'].array() for feat in feature_list]
        
        if len(mc_reco) ==0:
            mc_reco = np.stack(mcr,-1)
            mc_gen = np.stack(mcg,-1)
            reco_mask = tmp_file['passed_measurement_rec'].array() ==1.0            
            gen_mask =  tmp_file['passed_measurement_gen'].array() ==1.0
            mc_weights = tmp_file['gen_weight'].array()
            mc_weights[reco_mask==1.0]*= tmp_file['rec_weight'].array()[reco_mask==1.0]

        else:
            mc_reco = np.concatenate([mc_reco,np.stack(mcr,-1)],0)
            mc_gen =  np.concatenate([mc_gen,np.stack(mcg,-1)],0)
            reco_mask = np.concatenate([reco_mask, tmp_file['passed_measurement_rec'].array()==1.0],0)
            gen_mask =  np.concatenate([gen_mask, tmp_file['passed_measurement_gen'].array()==1.0],0)
            mcw = tmp_file['gen_weight'].array()
            mcw[tmp_file['passed_measurement_rec'].array()==1.0]*= tmp_file['rec_weight'].array()[tmp_file['passed_measurement_rec'].array()==1.0]
            mc_weights =  np.concatenate([mc_weights, mcw],0)
            
        
    #Missing entries are filled with nans
    mc_reco[np.isnan(mc_reco)] = 0.0
    mc_gen[np.isnan(mc_gen)] = 0.0
    
    
    #Now let's do the same thing with the 'data'. The main difference is that in data there are only events with reco_mask==True

    for sample in config['FILE_DATA']:
        file_path = os.path.join(base_path,sample)
        tmp_file = uproot.open(file_path)['AnalysisTree']
        # print(tmp_file.keys())
        data = [tmp_file[feat+'_rec'].array() for feat in feature_list]
        
        if len(data_reco) ==0:
            data_reco = np.stack(data,-1)[tmp_file['passed_measurement_rec'].array() ==1.0]
        else:
            data_reco = np.concatenate([data_reco,
                                        np.stack(data,-1)[tmp_file['passed_measurement_rec'].array() ==1.0]],0)
            
        
    #Missing entries are filled with nans
    mc_reco[np.isnan(mc_reco)] = 0.0
    mc_gen[np.isnan(mc_gen)] = 0.0
    data_reco[np.isnan(data_reco)] = 0.0

    #Lets convert the data from cartesian to cylindrical coordinates and reshape it
    mc_reco = convert_to_polar(mc_reco.reshape((-1,3,4)))
    mc_gen = convert_to_polar(mc_gen.reshape((-1,3,4)))
    data_reco = convert_to_polar(data_reco.reshape((-1,3,4)))


    return data_reco, mc_reco,mc_gen,reco_mask,gen_mask, mc_weights

def Plot_2D(sample,name,use_hist=True,weights=None):
    #cmap = plt.get_cmap('PiYG')
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad("white")
    # plt.rcParams['pcolor.shading'] ='nearest'

        
    def SetFig(xlabel,ylabel):
        fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(1, 1) 
        ax0 = plt.subplot(gs[0])
        ax0.yaxis.set_ticks_position('both')
        ax0.xaxis.set_ticks_position('both')
        ax0.tick_params(direction="in",which="both")    
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel(xlabel,fontsize=20)
        plt.ylabel(ylabel,fontsize=20)
        
        ax0.minorticks_on()
        return fig, ax0


    fig,ax = SetFig("x","y")

    
    if use_hist:
        if weights is None:
            weights = np.ones(sample.shape[0])
        im=plt.hist2d(sample[:,0],sample[:,1],
                      bins = 50,
                      range=[[-2,2],[-2,2]],
                      weights=weights,
                      cmap =cmap)
        cbar=fig.colorbar(im[3], ax=ax,label='Number of events')
    else:
        x=np.linspace(-2,2,50)
        y=np.linspace(-2,2,50)
        X,Y=np.meshgrid(x,y)
        im=ax.pcolormesh(X,Y,sample, cmap=cmap, shading='auto')
        fig.colorbar(im, ax=ax,label='Standard deviation')
        

    
    plot_folder='../plots'
    fig.savefig('{}/{}.pdf'.format(plot_folder,name))



def HistRoutine(feed_dict,xlabel='',ylabel='',reference_name='Geant4',logy=False,binning=None,label_loc='best',plot_ratio=True,weights=None,uncertainty=None):
    assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"
    
    fig,gs = SetGrid(ratio=plot_ratio) 
    ax0 = plt.subplot(gs[0])
    if plot_ratio:
        plt.xticks(fontsize=0)
        ax1 = plt.subplot(gs[1],sharex=ax0)

    
    if binning is None:
        binning = np.linspace(np.quantile(feed_dict[reference_name],0.0),np.quantile(feed_dict[reference_name],1),10)
        
    xaxis = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]
    reference_hist,_ = np.histogram(feed_dict[reference_name],bins=binning,density=True)
    for ip,plot in enumerate(feed_dict.keys()):
        if weights is not None:
            dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=plot,linestyle=line_style[plot],color=colors[plot],density=True,histtype="step",weights=weights[plot])
        else:
            dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=plot,linestyle=line_style[plot],color=colors[plot],density=True,histtype="step")
        
        if plot_ratio:
            if reference_name!=plot:
                ratio = np.ma.divide(dist,reference_hist).filled(0)
                ax1.plot(xaxis,ratio,color=colors[plot],marker='o',ms=10,lw=0,markerfacecolor='none',markeredgewidth=3)
                if uncertainty is not None:
                    for ibin in range(len(binning)-1):
                        xup = binning[ibin+1]
                        xlow = binning[ibin]
                        ax1.fill_between(np.array([xlow,xup]),
                                         uncertainty[ibin],-uncertainty[ibin], alpha=0.3,color='k')    
    if logy:
        ax0.set_yscale('log')

    ax0.legend(loc=label_loc,fontsize=16,ncol=1)        
    if plot_ratio:
        FormatFig(xlabel = "", ylabel = ylabel,ax0=ax0) 
        plt.ylabel('Ratio to Gen.')
        plt.axhline(y=1.0, color='r', linestyle='-',linewidth=1)
        # plt.axhline(y=10, color='r', linestyle='--',linewidth=1)
        # plt.axhline(y=-10, color='r', linestyle='--',linewidth=1)
        plt.ylim([0.8,1.2])
        plt.xlabel(xlabel)
    else:
        FormatFig(xlabel = xlabel, ylabel = ylabel,ax0=ax0) 
        
    return fig,ax0

