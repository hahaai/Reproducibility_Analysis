import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from scipy import stats
from scipy.io import loadmat
from scipy.stats import rankdata
from scipy.stats import spearmanr
import random



def spatial_corr_ridgeplot(base,outpath,pipelines_all,pairs2plot,namechangedict,fc_handle,simpleplot,corr_type):

    '''
    Function to prepare and, optionally, run the C-PAC workflow
    Parameters
    ----------
    base : string
        the base direcotory, the ROI data will be in base +'/ROI_Schaefer' + atlas + '/' + pipelinename
    pipelines : list of strings
        list of the pipelines to do correlaitons
    pairs2plot : list
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

    def cumulativeplott(df,outfile):
        roworder = ['0_3','3_4','0_4','1_3','2_4','2_3']


        kwargs = {'cumulative': True}
        sns.distplot(x, hist_kws=kwargs, kde_kws=kwargs)




    def ridgeplot(df,outfile):
        
        # handle soem constant 1 in df
        df=df.replace(1,(random.randrange(0,10)-5)/1000.0 + 1)
        
        
        #plt.subplots(figsize=(10,20))

        show_label=False
        # Initialize the FacetGrid object
        x=[]
        for i in np.unique(df['g']):
            x.append(np.median(df[df['g']==i]['x']))
        roworder=np.unique(df['g'])[np.argsort(x)[::-1]]

        # force the roworder to:
        roworder = ['0_3','3_4','0_4','1_3','2_4','2_3']
        #roworder=['0_4']

        # for colour and row name match
        with sns.plotting_context(font_scale=5):
            g1 = sns.FacetGrid(df, row="g", hue="g", aspect=len(pipelines)*(len(pipelines)-1)/2, height=2)
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
        #pal=pal[sortidx]
        with sns.plotting_context(font_scale=5):
            g = sns.FacetGrid(df, sharey=True, row="g", hue="g", row_order=roworder, aspect=2.5, height=4, palette=pal)
#           g = sns.FacetGrid(df, sharey=True, row="g", hue="g", row_order=roworder, aspect=len(pipelines)*(len(pipelines)-1)/4, height=2*10/(len(pipelines)*(len(pipelines)-1)/2), palette=pal)

            #g = sns.FacetGrid(df, row="g", hue="g", aspect=10, height=2, palette=pal)

        # Draw the densities in a few steps
        #g.map(sns.kdeplot, "x",hist=False)
        
        #g.map(sns.kdeplot, "x",shade=True)
        cumulativeplot=True
        shadeplot=False 
        lw_value = 5
        g.map(sns.kdeplot, "x",shade=shadeplot,color='red',cumulative=cumulativeplot,linewidth=lw_value) # minimal
        g.map(sns.kdeplot, "xx",shade=shadeplot,color='blue',cumulative=cumulativeplot,linewidth=lw_value) # minimal + filter
        g.map(sns.kdeplot, "xxx",shade=shadeplot,color='black',cumulative=cumulativeplot,linewidth=lw_value) # minimal + nuisance
        g.map(sns.kdeplot, "xxxx",shade=shadeplot,color='green',cumulative=cumulativeplot,linewidth=lw_value) # End-to-End



        #g.map(sns.kdeplot, df2['x'],shade=True,color='blue')
        lw_value=3
        g.map(plt.axvline, x=1, lw=lw_value, clip_on=False,color=(134/256.0,250/256.0,167/256.0))
        g.map(plt.axvline, x=0.9, lw=lw_value, clip_on=False,color=(128/256.0,171/256.0,69/256.0))
        g.map(plt.axvline, x=0.8, lw=lw_value, clip_on=False,color=(128/256.0,126/256.0,38/256.0))
        g.map(plt.axvline, x=0.7, lw=lw_value, clip_on=False,color=(246/256.0,192/256.0,66/256.0))
        g.map(plt.axvline, x=0.6, lw=lw_value, clip_on=False,color=(192/256.0,98/256.0,43/256.0))


        #g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw=.2)
        g.map(plt.axhline, y=0, lw=2, clip_on=False)


        # Define and use a simple function to label the plot in axes coordinates
        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .2, label, fontweight="bold", color='black',fontsize=35,
                    ha="left", va="center", transform=ax.transAxes)
            #ax.text(0, .2, label, fontweight="bold", color=color,fontsize=25,
            #        ha="left", va="center", transform=ax.transAxes)

        if show_label==True:
            g.map(label, "g")
        # Set the subplots to overlap
        #g.fig.subplots_adjust(hspace=-.6)

        # change the x axis label size
        ax = plt.gca()
        ax.tick_params(axis = 'both', which = 'major', labelsize = 35)

        # Remove axes details that don't play well with overlap
        g.set(xlim=(0, 1))
        if cumulativeplot:
            g.set(ylim=(0, 1))
        else:
            g.set(ylim=(0, 55))

        g.set_titles("")
        g.set(xlabel='')
        g.set(yticks=[])
        g.despine(bottom=True, left=True)
        g.savefig(outfile)
        return [x for _,x in sorted(zip(sortidx,default_rowname))]

    #plt.cla()
    #plt.subplots(figsize=(10,8))

    #for atlas in ['200','600','1000']:


    atlas='200'
    pair_idx=0
    for pair in pairs2plot:
        # get the pipelines only for the currect pair.
        pipelines = [x[nn] for nn in pair for x in pipelines_all]

        #num_idx=(len(pipelines)*(len(pipelines)-1))/2
        num_idx=len(pairs2plot)

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
            for xxx in [item for sublist in pipelines_all for item in sublist]:
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
                    locals()[pp] = spatial_correlation(corr1,corr2,locals()[pp],corr_type)

        

    value_median=pd.DataFrame(np.zeros((len(pairs2plot),5)), columns=['Pipelines','0_4','1_5','2_6','3_7'])
    value_std=pd.DataFrame(np.zeros((len(pairs2plot),5)), columns=['Pipelines','0_4','1_5','2_6','3_7']) # ['Pipelines','Minimal','Minimal_filter','Minimal_nuisance','End_to_end']

    value_quartile=pd.DataFrame(np.zeros((len(pairs2plot),17)), columns=['Pipelines','25_0_4','50_0_4','75_0_4','100_0_4','25_1_5','50_1_5','75_1_5','100_1_5','25_2_6','50_2_6','75_2_6','100_2_6','25_3_7','50_3_7','75_3_7','100_3_7']) # ['Pipelines','Minimal','Minimal_filter','Minimal_nuisance','End_to_end']


    #value_quartile=pd.DataFrame(np.zeros(((len(pipelines)*(len(pipelines)-1))/2,13)), columns=['Pipelines','25_Atlas200','50_Atlas200','75_Atlas200','100_Atlas200','25_Atlas600','50_Atlas600','75_Atlas600','100_Atlas600','25_Atlas1000','50_Atlas1000','75_Atlas1000','100_Atlas1000'])


    idx=0
    for i, j in [[0,4],[1,5],[2,6],[3,7]]:
        # get the pipelines only for the currect pair.
        num_idx=len(pairs2plot)
        color_palette = sns.color_palette("Paired",num_idx)
        

        df_all=pd.DataFrame(columns = ['x','g'])
        idx_pair=0
        for pair in pairs2plot:
            pipelines = [x[nn] for nn in pair for x in pipelines_all]
            print(idx)
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
                pn1=pn1.replace(key,namechangedict[key])
            #pn1=pn1.replace('cpac','CPAC:fmriprep')
            for key in namechangedict:
                pn2=pn2.replace(key,namechangedict[key])
            #pn2=pn2.replace('cpac','CPAC:fmriprep')

            print(pp1+str(np.median(pp))+str(np.std(pp)))
            value_median['Pipelines'][idx_pair] = str(pair)
            value_median[str(i)+'_'+str(j)][idx_pair] = np.median(pp)

            value_quartile['Pipelines'][idx_pair] = str(pair)
            for pct_val in [25,50,75, 100]:
                value_quartile[str(pct_val)+'_'+str(i)+'_'+str(j)][idx_pair] = np.percentile(pp, pct_val, interpolation = 'midpoint') 



            value_std['Pipelines'][idx_pair] = str(pair)
            value_std[str(i)+'_'+str(j)][idx_pair] = np.std(pp)


            #print('lenhg of pp')
            #print(len(pp))
            tmp=pd.DataFrame(pp, columns=['x'])
            #tmp['g']=pn1+' - '+pn2
            tmp['g']='_'.join([str(x) for x in pair])
            print(pn1+' - '+pn2)
            df_all=pd.concat([df_all,tmp])
            idx_pair += 1
        #ridgeplot(df_all,os.path.dirname(base) + '/figures/Ridgeplot_spatial_corr_'+corr_type+'_'+'-'.join(pipelines)+'_'+atlas+'.png')

    # put multipel atlas in one redge plot. 
        if idx==0:
            df_ridge =df_all
        elif idx==1:
            df_ridge['xx']=df_all['x']
        elif idx==2:
            df_ridge['xxx']=df_all['x']
        elif idx==3:
            df_ridge['xxxx']=df_all['x']
        idx += 1 

        pair_idx += 1
    #plotnameorder=ridgeplot(df_ridge,os.path.dirname(base) + '/figures/Ridgeplot_spatial_corr_'+corr_type+'_'+'-'.join(pipelines)+'_'+atlas+'.png')

    print(value_median)
    value_median.to_csv(os.path.dirname(base) + '/figures/Ridgeplot_spatial_corr_'+corr_type+'_'+atlas+'_Median.csv')
    value_std.to_csv(os.path.dirname(base) + '/figures/Ridgeplot_spatial_corr_'+corr_type+'_'+atlas+'_Std.csv')

    value_quartile.to_csv(os.path.dirname(base) + '/figures/Ridgeplot_spatial_corr_'+corr_type+'_'+atlas+'_quartile.csv')


    plotnameorder=ridgeplot(df_ridge,os.path.dirname(base) + '/figures/Ridgeplot_spatial_corr_'+corr_type+'_'+atlas+'.png')

    return plotnameorder

def ICC_ridgeplot(base,outpath,pipelines_all,pairs2plot,namechangedict,simpleplot,plotnameorder):
    # cpac_default_off_nuisance_ccs_ICC.csv

    def ridgeplot_test(df,outfile,plotnameorder):
        show_label=False
        #plt.subplots(figsize=(10,20))
        # Initialize the FacetGrid object
        x=[]
        for i in np.unique(df['g']):
            x.append(np.median(df[df['g']==i]['x']))
        roworder=np.unique(df['g'])[np.argsort(x)[::-1]]

        # for colour and row name match
        with sns.plotting_context(font_scale=5):
            g1 = sns.FacetGrid(df, row="g", hue="g", aspect=len(pipelines)*(len(pipelines)-1)/2, height=2)
        default_rowname=g1.row_names

        if plotnameorder:
            default_rowname=plotnameorder
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
            #g = sns.FacetGrid(df, row="g", hue="g", row_order=roworder, aspect=len(pipelines)*(len(pipelines)-1)/4, height=2*10/(len(pipelines)*(len(pipelines)-1)/2), palette=pal)
            g = sns.FacetGrid(df, sharey=True,row='g',hue="atlas", row_order=roworder, aspect=len(pipelines)*(len(pipelines)-1)/4, height=2*10/(len(pipelines)*(len(pipelines)-1)/2), palette=['black','red','blue'])

        # Draw the densities in a few steps
        #g.map(sns.kdeplot, "x",hist=False)
        
        #g.map(sns.kdeplot, "x",shade=True,color='red')
        g.map(sns.kdeplot, "x",shade=True)

        #.add_legend()


        #g.map(sns.kdeplot, df2['x'],shade=True,color='blue')
        lw_value=3
        g.map(plt.axvline, x=1, lw=lw_value, clip_on=False,color=(134/256.0,250/256.0,167/256.0))
        g.map(plt.axvline, x=0.9, lw=lw_value, clip_on=False,color=(128/256.0,171/256.0,69/256.0))
        g.map(plt.axvline, x=0.8, lw=lw_value, clip_on=False,color=(128/256.0,126/256.0,38/256.0))
        g.map(plt.axvline, x=0.7, lw=lw_value, clip_on=False,color=(246/256.0,192/256.0,66/256.0))
        g.map(plt.axvline, x=0.6, lw=lw_value, clip_on=False,color=(192/256.0,98/256.0,43/256.0))

        #g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw=.2)
        g.map(plt.axhline, y=0, lw=2, clip_on=False)


        # Define and use a simple function to label the plot in axes coordinates
        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .2, label, fontweight="bold", color='black',fontsize=35,
                    ha="left", va="center", transform=ax.transAxes)
            #ax.text(0, .2, label, fontweight="bold", color=color,fontsize=25,
            #        ha="left", va="center", transform=ax.transAxes)
        if show_label==True:
            g.map(label, "x")
        # Set the subplots to overlap
        #g.fig.subplots_adjust(hspace=-.6)

        # change the x axis label size
        ax = plt.gca()
        ax.tick_params(axis = 'both', which = 'major', labelsize = 35)

        # Remove axes details that don't play well with overlap
        g.set(xlim=(0, 1))
        g.set_titles("")
        g.set(xlabel='')
        g.set(yticks=[])
        g.despine(bottom=True, left=True)
        g.savefig(outfile)

    def ridgeplot(df,outfile):
        
        # handle soem constant 1 in df
        df=df.replace(1,(random.randrange(0,10)-5)/1000.0 + 1)
        
        
        #plt.subplots(figsize=(10,20))

        show_label=False
        # Initialize the FacetGrid object
        x=[]
        for i in np.unique(df['g']):
            x.append(np.median(df[df['g']==i]['x']))
        roworder=np.unique(df['g'])[np.argsort(x)[::-1]]

        # force the roworder to:
        roworder = ['0_3','3_4','0_4','1_3','2_4','2_3']


        # for colour and row name match
        with sns.plotting_context(font_scale=5):
            g1 = sns.FacetGrid(df, row="g", hue="g", aspect=len(pipelines)*(len(pipelines)-1)/2, height=2)
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
        #pal=pal[sortidx]
        with sns.plotting_context(font_scale=5):
            g = sns.FacetGrid(df, sharey=True, row="g", hue="g", row_order=roworder, aspect=2.5, height=4, palette=pal)
#           g = sns.FacetGrid(df, sharey=True, row="g", hue="g", row_order=roworder, aspect=len(pipelines)*(len(pipelines)-1)/4, height=2*10/(len(pipelines)*(len(pipelines)-1)/2), palette=pal)

            #g = sns.FacetGrid(df, row="g", hue="g", aspect=10, height=2, palette=pal)

        # Draw the densities in a few steps
        #g.map(sns.kdeplot, "x",hist=False)
        
        #g.map(sns.kdeplot, "x",shade=True)
        cumulativeplot=True
        shadeplot=False
        lw_value = 5
        g.map(sns.kdeplot, "x",shade=shadeplot,color='red',cumulative=cumulativeplot,linewidth=lw_value) # minimal
        g.map(sns.kdeplot, "xx",shade=shadeplot,color='blue',cumulative=cumulativeplot,linewidth=lw_value) # minimal + filter
        g.map(sns.kdeplot, "xxx",shade=shadeplot,color='black',cumulative=cumulativeplot,linewidth=lw_value) # minimal + nuisance
        g.map(sns.kdeplot, "xxxx",shade=shadeplot,color='green',cumulative=cumulativeplot,linewidth=lw_value) # End-to-End



        #g.map(sns.kdeplot, df2['x'],shade=True,color='blue')
        lw_value=3
        g.map(plt.axvline, x=1, lw=lw_value, clip_on=False,color=(134/256.0,250/256.0,167/256.0))
        g.map(plt.axvline, x=0.9, lw=lw_value, clip_on=False,color=(128/256.0,171/256.0,69/256.0))
        g.map(plt.axvline, x=0.8, lw=lw_value, clip_on=False,color=(128/256.0,126/256.0,38/256.0))
        g.map(plt.axvline, x=0.7, lw=lw_value, clip_on=False,color=(246/256.0,192/256.0,66/256.0))
        g.map(plt.axvline, x=0.6, lw=lw_value, clip_on=False,color=(192/256.0,98/256.0,43/256.0))


        #g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw=.2)
        g.map(plt.axhline, y=0, lw=2, clip_on=False)


        # Define and use a simple function to label the plot in axes coordinates
        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .2, label, fontweight="bold", color='black',fontsize=35,
                    ha="left", va="center", transform=ax.transAxes)
            #ax.text(0, .2, label, fontweight="bold", color=color,fontsize=25,
            #        ha="left", va="center", transform=ax.transAxes)

        if show_label==True:
            g.map(label, "g")
        # Set the subplots to overlap
        #g.fig.subplots_adjust(hspace=-.6)

        # change the x axis label size
        ax = plt.gca()
        ax.tick_params(axis = 'both', which = 'major', labelsize = 35)

        # Remove axes details that don't play well with overlap
        g.set(xlim=(0, 1))
        if cumulativeplot:
            g.set(ylim=(0, 1))
        else:
            g.set(ylim=(0, 55))

        g.set_titles("")
        g.set(xlabel='')
        g.set(yticks=[])
        g.despine(bottom=True, left=True)
        g.savefig(outfile)
        return [x for _,x in sorted(zip(sortidx,default_rowname))]



    value_median=pd.DataFrame(np.zeros((len(pairs2plot),5)), columns=['Pipelines','0_4','1_5','2_6','3_7'])
    value_std=pd.DataFrame(np.zeros((len(pairs2plot),5)), columns=['Pipelines','0_4','1_5','2_6','3_7']) # ['Pipelines','Minimal','Minimal_filter','Minimal_nuisance','End_to_end']

    value_quartile=pd.DataFrame(np.zeros((len(pairs2plot),17)), columns=['Pipelines','25_0_4','50_0_4','75_0_4','100_0_4','25_1_5','50_1_5','75_1_5','100_1_5','25_2_6','50_2_6','75_2_6','100_2_6','25_3_7','50_3_7','75_3_7','100_3_7']) # ['Pipelines','Minimal','Minimal_filter','Minimal_nuisance','End_to_end']


    idx=0
    atlas='200'
    for i, j in [[0,4],[1,5],[2,6],[3,7]]:
        # get the pipelines only for the currect pair.
        num_idx=len(pairs2plot)
        color_palette = sns.color_palette("Paired",num_idx)
        
        idx_pair = 0
        df_all=pd.DataFrame(columns = ['x','g'])
        for pair in pairs2plot:
            pipelines = [x[nn] for nn in pair for x in pipelines_all]
            #print(idx)
            p1=pipelines[i]
            p2=pipelines[j]

            p1=p1.replace('ABCD','abcd')
            p2=p2.replace('ABCD','abcd')
            pp1= outpath+ '/ICC_Schaefer' + atlas + '/' + p1 + '_' + p2 + '_ICC.csv'
            pp2= outpath+ '/ICC_Schaefer' + atlas + '/' + p2 + '_' + p1 + '_ICC.csv'
            #print(pp1)
            #print(pp2)

            if os.path.isfile(pp1):
                pp=pd.read_csv(pp1,header=None,names=['x'])
                #print(pp1)
            elif os.path.isfile(pp2):
                pp=pd.read_csv(pp2,header=None,names=['x'])
                #print(pp1)
            else:
                print(p1,p2,'NOT FOUND')
            if p1 == p2:
                pp['x']=1

            print(pp1+str(np.median(pp))+str(np.std(pp)))
            value_median['Pipelines'][idx_pair] = str(pair)
            value_median[str(i)+'_'+str(j)][idx_pair] = np.median(pp)

            value_std['Pipelines'][idx_pair] = str(pair)
            value_std[str(i)+'_'+str(j)][idx_pair] = np.std(pp)

            value_quartile['Pipelines'][idx_pair] = str(pair)
            for pct_val in [25,50,75, 100]:
                value_quartile[str(pct_val)+'_'+str(i)+'_'+str(j)][idx_pair] = np.percentile(pp, pct_val, interpolation = 'midpoint') 



            tmp=pd.DataFrame(pp, columns=['x'])
            #tmp['g']=pn1+' - '+pn2
            tmp['g']='_'.join([str(x) for x in pair])
            df_all=pd.concat([df_all,tmp])

            idx_pair += 1
        #ridgeplot(df_all,os.path.dirname(base) + '/figures/Ridgeplot_spatial_corr_'+corr_type+'_'+'-'.join(pipelines)+'_'+atlas+'.png')

    # put multipel atlas in one redge plot. 
        if idx==0:
            df_ridge =df_all
        elif idx==1:
            df_ridge['xx']=df_all['x']
        elif idx==2:
            df_ridge['xxx']=df_all['x']
        elif idx==3:
            df_ridge['xxxx']=df_all['x']
        idx += 1 

    print(value_median)
    value_median.to_csv(os.path.dirname(base) + '/figures/Ridgeplot_ICC_'+atlas+'_Median.csv')
    value_std.to_csv(os.path.dirname(base) + '/figures/Ridgeplot_ICC_'+atlas+'_Std.csv')

    value_quartile.to_csv(os.path.dirname(base) + '/figures/Ridgeplot_ICC_'+atlas+'_quartile.csv')

    ridgeplot(df_ridge,os.path.dirname(base) + '/figures/Ridgeplot_ICC_'+atlas+'.png')

### Example
'''
fc_handle=''
corr_type='spearman'
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

spatial_corr_ridgeplot(base,base.replace('ROI','figures'),pipelines,atlases,namechangedict,fc_handle,simpleplot,corr_type)
'''
