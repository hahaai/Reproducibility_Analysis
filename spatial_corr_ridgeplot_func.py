import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from scipy import stats
from scipy.io import loadmat
from scipy.stats import rankdata




def spatial_corr_ridgeplot(base,outpath,pipelines,atlases,namechangedict,fc_handle,simpleplot,corr_type):

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


    def ridgeplot(df,outfile):
        
        #plt.subplots(figsize=(10,20))


        # Initialize the FacetGrid object
        x=[]
        for i in np.unique(df['g']):
            x.append(np.median(df[df['g']==i]['x']))
        roworder=np.unique(df['g'])[np.argsort(x)[::-1]]

        # for colour and row name match
        with sns.plotting_context(font_scale=5):
            g1 = sns.FacetGrid(df, row="g", hue="g", aspect=10, height=2)
        default_rowname=g1.row_names

        sortidx=[]
        for i in default_rowname:
            sortidx.append(list(roworder).index(i))


        #pal = sns.cubehelix_palette(len(np.unique(df['g'])), rot=-.25, light=.7)
        #pal = np.asarray(sns.cubehelix_palette(len(np.unique(df['g'])), rot=.9, light=.7))
        #pal = np.asarray(sns.color_palette("purple", len(np.unique(df['g']))))
        pal = np.asarray(sns.cubehelix_palette(len(np.unique(df['g'])), start=.5, rot=-.75,light=0.7))

        #pal=pal[sortidx][::-1]
        pal=pal[::-1]
        pal=pal[sortidx]
        with sns.plotting_context(font_scale=5):
            g = sns.FacetGrid(df, row="g", hue="g", row_order=roworder, aspect=10, height=2, palette=pal)
            #g = sns.FacetGrid(df, row="g", hue="g", aspect=10, height=2, palette=pal)

        # Draw the densities in a few steps
        #g.map(sns.kdeplot, "x",hist=False)
        g.map(sns.kdeplot, "x",shade=True)
        g.map(plt.axvline, x=0.9, lw=2, clip_on=False)
        #g.map(plt.axvline, x=0.8, lw=1, clip_on=False)
        #g.map(plt.axvline, x=0.7, lw=1, clip_on=False)

        #g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw=.2)
        g.map(plt.axhline, y=0, lw=2, clip_on=False)


        # Define and use a simple function to label the plot in axes coordinates
        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .2, label, fontweight="bold", color='black',fontsize=25,
                    ha="left", va="center", transform=ax.transAxes)
            #ax.text(0, .2, label, fontweight="bold", color=color,fontsize=25,
            #        ha="left", va="center", transform=ax.transAxes)

        g.map(label, "x")
        # Set the subplots to overlap
        #g.fig.subplots_adjust(hspace=-.6)

        # change the x axis label size
        ax = plt.gca()
        ax.tick_params(axis = 'both', which = 'major', labelsize = 25)

        # Remove axes details that don't play well with overlap
        g.set(xlim=(0, 1))
        g.set_titles("")
        g.set(xlabel='')
        g.set(yticks=[])
        g.despine(bottom=True, left=True)
        g.savefig(outfile)

    #plt.cla()
    #plt.subplots(figsize=(10,8))

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

        #for i in range(1,31):
        basesub=25426
        for i in range(1,31):
            #if basesub+i == 25430:
            #    continue
            stop=0
            for xxx in pipelines:
                fodlercontent=os.listdir(base +'/ROI_Schaefer' + atlas + '/' + xxx)
                if str(i+basesub) not in str(fodlercontent):
                    print(xxx + 'not in')
                    print(i+basesub)
                    stop=1
            if stop==1:
                continue



            print(i)

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
                    locals()[pp] = spatial_correlation(corr1,corr2,locals()[pp])


        idx=0
        num_idx=(len(pipelines)*(len(pipelines)-1))/2
        color_palette = sns.color_palette("Paired",num_idx)
        
        df_all=test=pd.DataFrame(columns = ['x','g'])
        for i in range(0,len(pipelines)):
        #for i in range(0,1):
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
                    pn1=pn1.replace(key,namechangedict[key])
                #pn1=pn1.replace('cpac','CPAC:fmriprep')
                for key in namechangedict:
                    pn2=pn2.replace(key,namechangedict[key])
                #pn2=pn2.replace('cpac','CPAC:fmriprep')

                print(pp)

                tmp=pd.DataFrame(pp, columns=['x'])
                tmp['g']=pn1+' - '+pn2
                df_all=pd.concat([df_all,tmp])

        ridgeplot(df_all,os.path.dirname(base) + '/figures/Ridgeplot_spatial_corr_'+'-'.join(pipelines)+'_'+atlas+'.png')






### Example
'''
fc_handle=''
simpleplot=False
mulitple_atlas=False

if mulitple_atlas:
    atlases=['200','600','1000']
else:
    atlases=['200',]


base='/Users/leiai/projects/CPAC_fmriprep/Reliability_discriminibility/Finalizing/Minimal/ROI'

#pipelines=['cpac_fmriprep','abcd','fmriprep_1','fmriprep_MNI2004_2mm','dpabi','ccs']
#pipelines=['cpac_default_off_nuisance','abcd','fmriprep_1','fmriprep_MNI2004_2mm','dpabi','ccs']

pipelines=['cpac_fmriprep','cpac_default_off_nuisance','fmriprep_1','fmriprep_MNI2004_2mm','ccs']

# this is the name chang for the final plot.
namechangedict={'Upenn_cpacfmriprep':'fMRIPrep:UPenn',
            'abcd':'ABCD',
            'dpabi':'DPARSF',
            'fmriprep_fs':'fmriprep:freesurfer',
            'cpac_fnirt':'CPAC:fmriprep(fnirt)',
            'cpac_default_fnirt':'CPAC:default(fnirt)',
            'cpac_default_off_nuisance':'CPAC:Default',
            'cpac_fmriprep':'CPAC:fMRIPrep',
            'cpac_fmriprep_MNI2004':'CPAC:fmriprep',
            'ccs':'CCS',
            'fmriprep_1':'fMRIPrep',
            'fmriprep_MNI2004_2mm':'fMRIPrep(MNI2004-2mm)'
            }

spatial_corr_ridgeplot(base,base.replace('ROI','figures'),pipelines,atlases,namechangedict,fc_handle,simpleplot)
'''
