import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from scipy import stats
from scipy.io import loadmat
from scipy.stats import rankdata
from scipy.stats import spearmanr



def spatial_corr_plot(base,outpath,pipelines_list,atlases,namechangedict,fc_handle,simpleplot,corr_type):

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

    def load_multple_sess_data(datafolder,sub,sessions):
        if 'sess_ave' in locals():
            del sess_ave
        for ses in sessions:
            cpacfile=datafolder + '/sub-00' + str(sub) + ses + '.1D'
            #print(cpacfile)
            data=np.genfromtxt(cpacfile)
            if data.shape[0] != 295:
                idx= data.shape[0]-295
                data=data[idx:,]
            data=data.transpose()
            data_corr=np.corrcoef(data)
            tmp_corr_tri_tmp=upper_tri_indexing(data_corr)   
            if 'sess_ave' in locals():
                sess_ave=sess_ave+tmp_corr_tri_tmp
            else:
                sess_ave=tmp_corr_tri_tmp
        #print(sess_ave/len(sessions))
        return sess_ave/len(sessions)

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
            print(np.corrcoef(corr_a_new,corr_b_new)[0,1])
        return sc


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



    num_pair=len(pipelines_list) # this is number of rows of the plot

    duration10=[['a'],['j']]
    duration30=[['a','b','c'],['h','i','j']]
    duration50=[['a','b','c','d','e'],['f','g','h','i','j']]


    plt.cla()
    fig, axs = plt.subplots(num_pair,3,figsize=(15,12),sharex=True,dpi=600)
    plt.xlim(0, 1.1) 

    #plt.subplots(figsize=(10,8))
    for atlas in atlases:
        # loop through each row, pipelines pairs
        for p_idx in range(0,num_pair):
            print(p_idx)
            pipelines=pipelines_list[p_idx][0:2]
            pl_handle=pipelines_list[p_idx][2]
        
            # Loop through 10min, 30min, and 50min. columns
            col_indx=0
            for duration_col in [duration10,duration30,duration50]:
            #for duration_col in [duration10]:

                col_indx += 1
                print(duration_col)
                if pl_handle == 'different':
                    session_idx=[0,1]
                else:
                    session_idx=[0,0]
            
                #### Get each fiture done
                sess_all=[] # get all session used firstly, will check if have them all.
                for ii in session_idx:
                    for ses in duration_col[ii]:
                        sess_all.append(ses)

                for i in range(0,len(pipelines)):
                    for j in range(i+1,len(pipelines)):
                        p1=pipelines[i]
                        p2=pipelines[j]
                        pp='sc_' + p1 + '_' + p2
                        locals()[pp]=[]

                basesub=25426
                for isub in range(1,31):
                    if basesub+isub == 25430:
                        continue
                    stop=0
                    for xxx in pipelines:
                        fodlercontent=os.listdir(base +'/ROI_Schaefer' + atlas + '/' + xxx)
                        for ses in sess_all:
                            if (str(isub+basesub) + ses) not in str(fodlercontent):
                                print(isub+basesub)
                                stop=1
                    if stop==1:
                        continue





                    # same session and different session will be different:
                    for pl, ii in zip(pipelines, session_idx):
                        datafolder = base +'/ROI_Schaefer' + atlas + '/' + pl

                        # load the data - need to average sessions.
                        tmp_corr_tri=load_multple_sess_data(datafolder,str(isub+basesub),duration_col[ii])

                        if pl+'_corr_tri' in locals():
                            locals()[pl+'_1_corr_tri']=tmp_corr_tri
                        else:
                            locals()[pl+'_corr_tri']=tmp_corr_tri


                    ### do correlaiton between pipelines
                    num_idx=(len(pipelines)*(len(pipelines)-1))/2
                    color_palette = sns.color_palette("Paired",num_idx)
                    for i in range(0,len(pipelines)):
                        for j in range(i+1,len(pipelines)):
                            p1=pipelines[i]
                            p2=pipelines[j]

                            corr1= locals()[p1 + '_corr_tri']
                            if p1 == p2:
                                corr2= locals()[p2 + '_1_corr_tri']
                            else:
                                corr2= locals()[p2 + '_corr_tri']
                            pp='sc_' + p1 + '_' + p2
                            #print(pp)
                            #print(corr1)
                            #print(corr2)
                            locals()[pp] = spatial_correlation(corr1,corr2,locals()[pp],corr_type)

                    for pl, ii in zip(pipelines, session_idx):
                        if pl+'_1_corr_tri' in locals():
                            del locals()[pl+'_1_corr_tri']
                        if pl+'_corr_tri' in locals():
                            del locals()[pl+'_corr_tri']
                # here
            
                if simpleplot == True:
                    plotrange=1
                else:
                    plotrange=len(pipelines)
                for i in range(0,plotrange):
                    for j in range(i+1,len(pipelines)):
                        p1=pipelines[i]
                        p2=pipelines[j]
                        pp1='sc_' + p1 + '_' + p2
                        pp2='sc_' + p2 + '_' + p1
                        if pp1 in locals():
                            pp = locals()[pp1]
                            #print(pp1)
                        elif pp2 in locals():
                            pp = locals()[pp2]
                            #print(pp2)
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
                            g=sns.distplot(pp,color= 'black',hist=False,ax=axs[p_idx,col_indx-1],axlabel='')
                        elif atlas == '600':
                            g=sns.distplot(pp,color= 'black',hist=False, kde_kws={'linestyle':'--'},ax=axs[p_idx,col_indx-1],axlabel='')
                        else:
                            g=sns.distplot(pp,color= 'black',hist=False, kde_kws={'linestyle':':'},ax=axs[p_idx,col_indx-1],axlabel='')



    if fc_handle == '':
        plt.savefig(outpath + '/Mult_session_spatial_corr_'+corr_type+'.png')
    else:
        plt.savefig(outpath + '/Mult_session_spatial_corr_'+corr_type+'_'+fc_handle+'.png')


    '''
    ## plot
    fig1,axn1 = plt.subplots(1, 3, sharex=True, sharey=True,figsize=(9,3))
    sns_plot=sns.distplot
    (corr,square=True,vmax=1,vmin=-0.5,ax=axn.flat[idx+3],cmap='coolwarm')
    axn.flat[idx+3].set_title(sub)
    '''



####################################
#Example to call the functions
#pipelines=['cpac_fmriprep_MNI2004','abcd','fmriprep_MNI2004_2mm','Upenn_cpacfmriprep','dpabi','ccs']
pipelines_list=[
    ['cpac_default','cpac_default','different'],
    ['cpac_FX','cpac_FX','different'],
    ['XCP_fmriprep','XCP_fmriprep','different'],
    ['XCP_fmriprep','cpac_default','different'],
    ['XCP_fmriprep','cpac_FX','different'],
    ['XCP_fmriprep','cpac_default','same'],
    ['XCP_fmriprep','cpac_FX','same'],
    ['XCP_fmriprep','XCP_fmriprep','same'],
    ['cpac_default','cpac_default','same'],
    ['cpac_FX','cpac_FX','same']
    ]
'''
pipelines_list=[
    ['XCP_fmriprep','cpac_default','different'],
    ['XCP_fmriprep','cpac_FX','different'],
    ['XCP_fmriprep','cpac_FX','same']
    ]
pipelines_list=[
    ['XCP_fmriprep','cpac_FX','same']    ]
'''

fc_handle=''
corr_type='pearson'
simpleplot=False
atlases=['200',]
base='/Users/leiai/projects/CPAC_fmriprep/Reliability_discriminibility/Finalizing/All_sessions/ROI'


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


spatial_corr_plot(base,base.replace('ROI','figures'),pipelines_list,atlases,namechangedict,fc_handle,simpleplot,corr_type)


