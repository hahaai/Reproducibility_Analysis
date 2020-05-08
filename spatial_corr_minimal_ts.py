import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from scipy import stats
from scipy.io import loadmat
from scipy.stats import rankdata


# rsync -r  --verbose --ignore-existing lai@lisa.childmind.org:/data3/cnl/fmriprep/Lei_working/Finalizing/Minimal/ROI/ /Users/leiai/Dropbox/Work\&Study/projects/CPAC_fmriprep/Reliability_discriminibility/Finalizing/Minimal/ROI/


#fc_handle='Scale'
#fc_handle='Ranking'

fc_handle=''
mulitple_atlas=False

if mulitple_atlas:
    atlases=['200','600','1000']
else:
    atlases=['200',]


base='/Users/leiai/projects/CPAC_fmriprep/Reliability_discriminibility/Finalizing/Minimal/ROI'

pipelines=['cpac_default_off_nuisance','cpac_fmriprep','abcd','fmriprep_1','Upenn_cpacfmriprep','dpabi','CCS']

#pipelines=['cpac_default_off_nuisance','cpac_fmriprep']




# this is the name chang for the final plot.
namechange={'Upenn_cpacfmriprep':'fmriprep:Upenn',
            'abcd':'ABCD',
            'dpabi':'DPARSFA',
            'fmriprep_fs':'fmriprep:freesurfer',
            'cpac_fnirt':'CPAC:fmriprep(fnirt)',
            'cpac_default_fnirt':'CPAC:default(fnirt)',
            'cpac_default_off_nuisance':'CPAC:default',
            'cpac_fmriprep':'CPAC:fmriprep',
            'cpac_fmriprep_MNI2004':'CPAC:fmriprep'
            }


# /data3/cnl/fmriprep/Lei_working/CPAC_XCP/CCS_CPAC_XCP/CPAC_default/sub-0025427_ses-1_bandpassed_demeaned_filtered_antswarp_cc200.1D

# the order of ccs subjects correspondign to cpac nad ndmg
# cpac: 1-2,    3-14,   15-30
# ccs : 1-2,    19-30,  3-18

#\u221a/data2/Projects/Lei/ndmg/Resting_Preprocessing/hnu/ndmg_out/func/roi-timeseries/CPAC200_res-2x2x2/sub-0025427_task-rest_bold_CPAC200_res-2x2x2_variant-mean_timeseries.npz
def upper_tri_masking(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:,None] < r
    return A[mask]

def upper_tri_indexing(A):
    m = A.shape[0]
    r,c = np.triu_indices(m,1)
    return A[r,c]

### creat empty matrix





#for atlas in ['200','600','1000']:
for atlas in atlases:

    num_idx=(len(pipelines)*(len(pipelines)-1))/2
    color_palette = sns.color_palette("Paired",num_idx)
    for i in range(0,len(pipelines)):
        for j in range(i+1,len(pipelines)):
            p1=pipelines[i]
            p2=pipelines[j]
            pp='sc_' + p1 + '_' + p2
            locals()[pp]=[]

    ## loop through pipeline pairs
    idx=0
    num_idx=(len(pipelines)*(len(pipelines)-1))/2
    color_palette = sns.color_palette("Paired",num_idx)
    
    for ii in range(0,len(pipelines)):
    #for i in range(0,1):
        for j in range(ii+1,len(pipelines)):
            plt.cla()
            plt.subplots(figsize=(10,8))

            p1=pipelines[ii]
            p2=pipelines[j]
            p1path=base +'/ROI_Schaefer' + atlas  + '/' + p1
            p2path=base +'/ROI_Schaefer' + atlas  + '/' + p2
            print(p1 + ' and ' + p2)
            #for i in range(1,31):
            basesub=25426
            for i in range(1,31):
                #if basesub+i == 25430:
                #    continue
                stop=0
                for xxx in ['p1path','p2path']:
                    fodlercontent=os.listdir(locals()[xxx])
                    if str(i+basesub) not in str(fodlercontent):
                        print(i+basesub)
                        stop=1
                if stop==1:
                    continue
                #print(i)

                ### p1
                cpacfile=p1path + '/sub-00' + str(basesub+i) + 'a.1D'
                data=np.genfromtxt(cpacfile)
                if data.shape[0] != 295:
                    idx= data.shape[0]-295
                    data=data[idx:,]
                data_p1=data.transpose()

                ### p2
                cpacfile=p2path + '/sub-00' + str(basesub+i) + 'a.1D'
                data=np.genfromtxt(cpacfile)
                if data.shape[0] != 295:
                    idx= data.shape[0]-295
                    data=data[idx:,]
                data_p2=data.transpose()

                data_corr=[]
                for roi_idx in range(0,data_p1.shape[0]):
                    tmp1=data_p1[roi_idx,]
                    tmp2=data_p2[roi_idx,]
                    data_corr.append(np.corrcoef(tmp1,tmp2)[0,1])

                sns.distplot(data_corr,axlabel='',hist=False)
                sns.set(font_scale=1.5)
            plt.xlim(0, 1)             
            plt.savefig(os.path.dirname(base) + '/figures_ts/TS__corr_'+p1 + '_' + p2 + '_'+atlas+'.png')
         


