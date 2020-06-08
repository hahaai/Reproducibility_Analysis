import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from scipy import stats
from scipy.io import loadmat
from scipy.stats import rankdata
from scipy.stats import spearmanr



def spatial_corr_plot(base,outpath,pipelines1,pipelines2,pipelines3,pipeines4,atlases,namechangedict,fc_handle,simpleplot,corr_type):

    '''
    Function to prepare and, optionally, run the C-PAC workflow
    Parameters
    ----------
    base : string
        the base direcotory, the ROI data will be in base +'/ROI_Schaefer' + atlas + '/' + pipelinename
    pipelines : list of strings
        list of the pipelines to do correlaitons
    atlases : list
        list of atlases to ues, for example ['200','600','1000']
    namechangedict : dictionary
        keys are the pipeline names, values are the name to change to
    fc_handle : string
        how to hand the spatial correlation, it can be '','Scale','Ranking'
    simpleplot : boolean
        flag to indicate the combination of pipelines, True only plot the correlation betwene the first pipeline and the rest.
    corr_type: string
        which correlation to do: concordance, spearman, or pearson
    Returns:
        None, but save figure out.
    -------
    workflow : 
    '''


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

    # concordance_correlation_coefficient from https://github.com/stylianos-kampakis/supervisedPCA-Python/blob/master/Untitled.py
    def concordance_correlation_coefficient(y_true, y_pred,
                           sample_weight=None,
                           multioutput='uniform_average'):
        """Concordance correlation coefficient.
        The concordance correlation coefficient is a measure of inter-rater agreement.
        It measures the deviation of the relationship between predicted and true values
        from the 45 degree angle.
        Read more: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
        Original paper: Lawrence, I., and Kuei Lin. "A concordance correlation coefficient to evaluate reproducibility." Biometrics (1989): 255-268.  
        Parameters
        ----------
        y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
            Ground truth (correct) target values.
        y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
            Estimated target values.
        Returns
        -------
        loss : A float in the range [-1,1]. A value of 1 indicates perfect agreement
        between the true and the predicted values.
        Examples
        --------
        >>> from sklearn.metrics import concordance_correlation_coefficient
        >>> y_true = [3, -0.5, 2, 7]
        >>> y_pred = [2.5, 0.0, 2, 8]
        >>> concordance_correlation_coefficient(y_true, y_pred)
        0.97678916827853024
        """
        cor=np.corrcoef(y_true,y_pred)[0][1]
        
        mean_true=np.mean(y_true)
        mean_pred=np.mean(y_pred)
        
        var_true=np.var(y_true)
        var_pred=np.var(y_pred)
        
        sd_true=np.std(y_true)
        sd_pred=np.std(y_pred)
        
        numerator=2*cor*sd_true*sd_pred
        
        denominator=var_true+var_pred+(mean_true-mean_pred)**2

        return numerator/denominator




    plt.cla()
    fig, axs = plt.subplots(4,4,figsize=(15,12),sharex=True,sharey=True,dpi=600)
    plt.xlim(0, 1) 
    plt.ylim(0, 33) 


    base_orig=base
    ################################################ pipeline 1

    def plot_symmary(pipelines,base,axs_indx):
        for atlas in atlases:

            num_idx=(len(pipelines)*(len(pipelines)-1))/2
            color_palette = sns.color_palette("Paired",num_idx)
            for i in range(0,len(pipelines)):
                for j in range(i+1,len(pipelines)):
                    p1=pipelines[i]
                    p2=pipelines[j]
                    pp='sc_' + p1 + '_' + p2
                    locals()[pp]=[]
            basesub=25426
            for i in range(1,31):
                if basesub+i == 25430:
                    continue
                stop=0
                for xxx in pipelines:
                    fodlercontent=os.listdir(base +'/ROI_Schaefer' + atlas + '/' + xxx)
                    if str(i+basesub) not in str(fodlercontent):
                        print(i+basesub)
                        stop=1
                if stop==1:
                    continue

                print(i)

                ### NEW dpabi
                # read ROI first, there are 400 ROIS, the first 200 is cc200, the second 200 is sheafle.
                # ROISignals_sub-0025430

                # put them all together, load each pipeline file and calcuate calrelaiton and give it a different name.
                for pl in pipelines:
                    datafolder = base +'/ROI_Schaefer' + atlas + '/' + pl
                    # cpacdefault24
                    #/data3/cnl/fmriprep/Lei_working/CPAC_XCP/CPAC_aggre_output/sub-0025427a_ses-1_bandpassed_demeaned_filtered_antswarp_cc200.1D
                    cpacfile=datafolder + '/sub-00' + str(basesub+i) + 'a.1D'
                    data=np.genfromtxt(cpacfile)
                    if data.shape[0] != 295:
                        idx= data.shape[0]-295
                        data=data[idx:,]
                    data=data.transpose()
                    data_corr=np.corrcoef(data)
                    tmp_corr_tri=upper_tri_indexing(data_corr)
                    locals()[pl+'_corr_tri']=tmp_corr_tri


                
               # (df.a - df.a.mean())/df.a.std(ddof=0)
                def spatial_correlation(corr_a,corr_b,sc,corr_type):
                    if fc_handle=='Scale':
                        print('here')
                        corr_a[np.isnan(corr_a)]=0
                        corr_b[np.isnan(corr_b)]=0
                        corr_a= (corr_a - corr_a.mean())/corr_a.std(ddof=0)
                        corr_b= (corr_b - corr_b.mean())/corr_b.std(ddof=0)
                    if fc_handle == 'Ranking':
                        corr_a = rankdata(corr_a)
                        corr_b = rankdata(corr_b)
                    x=np.isnan(corr_a) | np.isnan(corr_b)
                    corr_a_new=corr_a[~x]
                    corr_b_new=corr_b[~x]
                    if corr_type == "spearman":
                        sc.append(spearmanr(corr_a_new,corr_b_new)[0])
                    elif corr_type == "concordance":
                        sc.append(concordance_correlation_coefficient(corr_a_new,corr_b_new))
                    else:
                        sc.append(np.corrcoef(corr_a_new,corr_b_new)[0,1])
                    return sc
    # 



                ### do correlaiton between pipelines
                num_idx=(len(pipelines)*(len(pipelines)-1))/2
                color_palette = sns.color_palette("Paired",num_idx)
                for i in range(0,len(pipelines)):
                    for j in range(i+1,len(pipelines)):
                        p1=pipelines[i]
                        corr1= locals()[p1 + '_corr_tri']
                        p2=pipelines[j]
                        corr2= locals()[p2 + '_corr_tri']
                        pp='sc_' + p1 + '_' + p2
                        locals()[pp] = spatial_correlation(corr1,corr2,locals()[pp],corr_type)


            idx=0
            num_idx=(len(pipelines)*(len(pipelines)-1))/2
            color_palette = sns.color_palette("Paired",num_idx)
            
            if simpleplot == True:
                plotrange=1
            else:
                plotrange=len(pipelines)
            for i in range(0,plotrange):
                for j in range(i+1,len(pipelines)):
                    print(idx)
                    p1=pipelines[i]
                    p2=pipelines[j]
                    pp1='sc_' + p1 + '_' + p2
                    pp2='sc_' + p2 + '_' + p1
                    if pp1 in locals():
                        pp = locals()[pp1]
                        print(pp1)
                    elif pp2 in locals():
                        pp = locals()[pp2]
                        print(pp2)
                    #pn1=p1.replace('newcpac','cpac:xcp').replace('defaultcpac','cpac:default')
                    #pn2=p2.replace('newcpac','cpac:xcp').replace('defaultcpac','cpac:default')
                    pn1=p1
                    pn2=p2
                    for key in namechangedict:
                        if pn1 == key:
                            pn1 = namechangedict[key]
                        #pn1=pn1.replace(key,namechangedict[key])
                        #pn1=re.sub(key,namechangedict[key],pn1)

                    pn1=pn1.replace('cpac','CPAC:fmriprep')
                    for key in namechangedict:
                        #pn2=re.sub(key,namechangedict[key],pn2)
                        if pn2 == key:
                            pn2 = namechangedict[key]
                    #pn2=pn2.replace('cpac','CPAC:fmriprep')
    # 
                    print(pp)
                    if atlas == '200':
                        g=sns.distplot(pp,color= 'black',hist=False,ax=axs[axs_indx,j-1])
                    elif atlas == '600':
                        g=sns.distplot(pp,color= 'black',hist=False, kde_kws={'linestyle':'--'},ax=axs[axs_indx,j-1])
                    else:
                        g=sns.distplot(pp,color= 'black',hist=False, kde_kws={'linestyle':':'},ax=axs[axs_indx,j-1])
                    idx += 1



    # add plots for all  4 pipelines
    plot_symmary(pipelines1,base.replace('Full','Minimal'),0)
    plot_symmary(pipelines2,base,1)
    plot_symmary(pipelines3,base,2)
    plot_symmary(pipelines4,base,3)


    if fc_handle == '':
        plt.savefig(outpath + '/Summary_spatial_corr_'+corr_type+'.png')
    else:
        plt.savefig(outpath + '/Summary_spatial_corr_'+corr_type+'_'+fc_handle+'.png')


    '''
    ## plot
    fig1,axn1 = plt.subplots(1, 3, sharex=True, sharey=True,figsize=(9,3))
    sns_plot=sns.distplot
    (corr,square=True,vmax=1,vmin=-0.5,ax=axn.flat[idx+3],cmap='coolwarm')
    axn.flat[idx+3].set_title(sub)
    '''





#Example to call the functions

pipelines1=['fmriprep_1','cpac_default_off_nuisance','cpac_fmriprep','cpac_default_off_nuisance','cpac_fmriprep']

# summary figure - only filter (minimal+filter)
pipelines2=['XCP_fmriprep_tmp','cpac_default_off_nuisance_with_filt','cpac_fmriprep_with_filt','XCP_cpac_default_tmp','XCP_cpac_FX_tmp']

# summary figure - only nuisance (minimal+nuisance)
pipelines3=['XCP_fmriprep_reg','cpac_default_filt_0_nuisance_1','cpac_FX_no_filt','XCP_cpac_default_reg','XCP_cpac_FX_reg']

# the main figure for end-to-end processing
pipelines4=['xcp_on_fmriprep','cpac_default_24','cpac_3dvolreg_36','xcp_on_cpac_default','xcp_on_cpac']




fc_handle=''
corr_type='pearson'
simpleplot=True
atlases=['200',]
base='/Users/leiai/projects/CPAC_fmriprep/Reliability_discriminibility/Finalizing/Full/ROI'


# this is the name chang for the final plot.
namechangedict={'cpac_fmriprep':'CPAC:fMRIPrep',
            #'cpac_fmriprep':'CPAC:fMRIPrep(MNI2009-3.4mm)',
            'cpac_default_no_nuisance':'CPAC:Default',
            'cpac_default_off_nuisance':'CPAC:Default',
            'cpac_default_no_nuisance_mni2009':'CPAC:Default(MNI2009)',
            'cpac_fmriprep_mni2004':'CPAC:fMRIPrep(MNI2004-2mm)',
            'fmriprep':'fMRIPrep',
            'cpac_fmriprep_anat_mask':'CPAC:fMRIPrep(Default Anat Mask)',
            'cpac_fmriprep_func_mask':'CPAC:fMRIPrep(Default Func Mask)',
            'cpac_fmriprep_coreg':'CPAC:fMRIPrep(Default Coreg)',
            'cpac_fmriprep_N4':'CPAC:fMRIPrep(No N4)',
            'cpac_fmriprep_stc':'CPAC:fMRIPrep(No STC)',
            'cpac_FX_no_filt':'CPAC:fMRIPrep(Nuisance)',
            'cpac_fmriprep_with_filt':'CPAC:fMRIPrep(Filter)',
            'cpac_3dvolreg_36':'CPAC:fMRIPrep(Nuisance + Filter)',
            'cpac_fmriprep_MNI2004_34mm':'CPAC:fMRIPrep(MNI2004-3.4mm)',
            'cpac_fmriprep_MNI2009_2mm':'CPAC:fMRIPrep(MNI2009-2mm)',
            'cpac_fmriprep_MNI2004':'CPAC:fMRIPrep(MNI2004-2mm)',
            'fmriprep_MNI2009_2mm':'fMRIPrep(MNI2009-2mm)',
            'fmriprep_MNI2004_2mm':'fMRIPrep(MNI2004-2mm)'
            }


spatial_corr_plot(base,base.replace('ROI','figures'),pipelines1,pipelines2,pipelines3,pipelines4,atlases,namechangedict,fc_handle,simpleplot,corr_type)



