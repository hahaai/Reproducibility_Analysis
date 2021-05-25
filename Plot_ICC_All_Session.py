############################################################################################################################################################################
## load spatial correlaiotn and plot 
import numpy as np, scipy.stats as st
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import random

axis_tick_size=22
lw_value=2
for corr_type in ['pearson']:


    num_pair=16
    plt.cla()
    fig, axs = plt.subplots(num_pair,3,figsize=(15,12),sharex=True,dpi=400)



    def plot_mean_and_CI(mean, lb, ub, color_mean=None, color_shading=None):
        # plot the shaded range of the confidence intervals
        plt.fill_between(range(len(mean)), ub, lb,
                         color=color_shading, alpha=.5)
        # plot the mean on top
        plt.plot(mean, color_mean)
        plt.show()


    def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
        #x_grid = np.linspace(-4.5, 3.5, 1000)
        """Kernel Density Estimation with Scipy"""
        # Note that scipy weights its bandwidth by the covariance of the
        # input data.  To make the results comparable to the other methods,
        # we divide the bandwidth by the sample standard deviation here.
        kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
        return kde.evaluate(x_grid)


    datain='./ICC_Plot'
    # Random-0_Pipeline-0_Ses_1.txt



    def re_sample(x,y,x_grid):
        y_out=np.linspace(0, 0, len(x_grid))

        stop=0
        for jj in range((len(x)-1)):
            for ii in range(len(x_grid)):
                grid=x_grid[ii]
                if grid >= x[jj] and grid <= x[jj+1]:
                    y_out[ii]=y[jj]
                if x[jj]== x[jj+1] and x[jj]==1 and stop == 0:
                    if grid >= x[jj]:
                        y_out[ii]=y[jj]  
                        stop=1
        return y_out




    ######## a new way to plot, combine package rows. not same and test-retest/same
    num_pair=16
    binnum=2000
    x_grid=np.linspace(0, 1.1, binnum)

    plt.cla()
    fig, axs = plt.subplots(6,3,figsize=(15,12),sharex=True,sharey='row',dpi=600)
    #plt.subplots_adjust(left=0, bottom=0, right=0.1, top=0.1)
    plt.tight_layout()
    #plt.xlabel('', fontsize=24)
    #plt.ylabel('', fontsize=24)


    for pl in range(0,num_pair):
        print(pl)
        for ses in range(1,4):
            print('ses is ' + str(ses))
            dataall=np.loadtxt(datain+'/Random_All'+'_Pipeline-'+str(pl)+'_Ses_'+str(ses)+'.txt')

            datamean=[]
            data_lb=[]
            data_up=[]
            for ii in range(0,dataall.shape[1]):
                    pdf=dataall[:,ii]                
                    a,b=st.t.interval(0.95, len(pdf)-1, loc=np.mean(pdf), scale=st.sem(pdf))
                    #print(str(a)+'_'+str(b))
                    if np.isnan(a):
                        a=0
                    if np.isnan(b):
                        b=0
                    data_lb.append(a)
                    data_up.append(b)
                    datamean.append(np.mean(pdf))

            if pl <=2:
                if pl==0:
                    color_plot='red'
                if pl==1:
                    color_plot='blue'
                if pl==2:
                    color_plot='green'
                axs[0, (ses-1)].fill_between(x_grid, data_lb,data_up,color=color_plot)
                axs[0, (ses-1)].plot(x_grid, datamean,linewidth=0.3,color='black')
                axs[0,(ses-1)].tick_params(axis='both', which='major', labelsize=axis_tick_size)

                axs[0, (ses-1)].axvline(x=1, lw=lw_value, clip_on=False,color=(134/256.0,250/256.0,167/256.0))
                axs[0, (ses-1)].axvline(x=0.9, lw=lw_value, clip_on=False,color=(128/256.0,171/256.0,69/256.0))
                axs[0, (ses-1)].axvline(x=0.8, lw=lw_value, clip_on=False,color=(128/256.0,126/256.0,38/256.0))
                axs[0, (ses-1)].axvline(x=0.7, lw=lw_value, clip_on=False,color=(246/256.0,192/256.0,66/256.0))
                axs[0, (ses-1)].axvline(x=0.6, lw=lw_value, clip_on=False,color=(192/256.0,98/256.0,43/256.0))


            if pl >2 and pl <= 5:
                if pl==3:
                    color_plot='red'
                if pl==4:
                    color_plot='blue'
                if pl==5:
                    color_plot='green'
                axs[1, (ses-1)].fill_between(x_grid, data_lb,data_up,color=color_plot)
                axs[1, (ses-1)].plot(x_grid, datamean,linewidth=0.3,color='black')
                axs[1,(ses-1)].tick_params(axis='both', which='major', labelsize=axis_tick_size)

                axs[1, (ses-1)].axvline(x=1, lw=lw_value, clip_on=False,color=(134/256.0,250/256.0,167/256.0))
                axs[1, (ses-1)].axvline(x=0.9, lw=lw_value, clip_on=False,color=(128/256.0,171/256.0,69/256.0))
                axs[1, (ses-1)].axvline(x=0.8, lw=lw_value, clip_on=False,color=(128/256.0,126/256.0,38/256.0))
                axs[1, (ses-1)].axvline(x=0.7, lw=lw_value, clip_on=False,color=(246/256.0,192/256.0,66/256.0))
                axs[1, (ses-1)].axvline(x=0.6, lw=lw_value, clip_on=False,color=(192/256.0,98/256.0,43/256.0))


            if pl >5 and pl <= 7:
                if pl==6:
                    color_plot='red'
                if pl==7:
                    color_plot='blue'
                axs[2, (ses-1)].fill_between(x_grid, data_lb,data_up,color=color_plot)
                axs[2, (ses-1)].plot(x_grid, datamean,linewidth=0.3,color='black')
                axs[2,(ses-1)].tick_params(axis='both', which='major', labelsize=axis_tick_size)

                axs[2, (ses-1)].axvline(x=1, lw=lw_value, clip_on=False,color=(134/256.0,250/256.0,167/256.0))
                axs[2, (ses-1)].axvline(x=0.9, lw=lw_value, clip_on=False,color=(128/256.0,171/256.0,69/256.0))
                axs[2, (ses-1)].axvline(x=0.8, lw=lw_value, clip_on=False,color=(128/256.0,126/256.0,38/256.0))
                axs[2, (ses-1)].axvline(x=0.7, lw=lw_value, clip_on=False,color=(246/256.0,192/256.0,66/256.0))
                axs[2, (ses-1)].axvline(x=0.6, lw=lw_value, clip_on=False,color=(192/256.0,98/256.0,43/256.0))

            if pl >7 and pl <= 10:
                if pl==8:
                    color_plot='red'
                if pl==9:
                    color_plot='blue'
                if pl==10:
                    color_plot='green'
                axs[3, (ses-1)].fill_between(x_grid, data_lb,data_up,color=color_plot)
                axs[3, (ses-1)].plot(x_grid, datamean,linewidth=0.3,color='black')
                axs[3,(ses-1)].tick_params(axis='both', which='major', labelsize=axis_tick_size)

                axs[3, (ses-1)].axvline(x=1, lw=lw_value, clip_on=False,color=(134/256.0,250/256.0,167/256.0))
                axs[3, (ses-1)].axvline(x=0.9, lw=lw_value, clip_on=False,color=(128/256.0,171/256.0,69/256.0))
                axs[3, (ses-1)].axvline(x=0.8, lw=lw_value, clip_on=False,color=(128/256.0,126/256.0,38/256.0))
                axs[3, (ses-1)].axvline(x=0.7, lw=lw_value, clip_on=False,color=(246/256.0,192/256.0,66/256.0))
                axs[3, (ses-1)].axvline(x=0.6, lw=lw_value, clip_on=False,color=(192/256.0,98/256.0,43/256.0))



            if pl >10 and pl <= 13:
                if pl==11:
                    color_plot='red'
                if pl==12:
                    color_plot='blue'
                if pl==13:
                    color_plot='green'
                axs[4, (ses-1)].fill_between(x_grid, data_lb,data_up,color=color_plot)
                axs[4, (ses-1)].plot(x_grid, datamean,linewidth=0.3,color='black')
                axs[4,(ses-1)].tick_params(axis='both', which='major', labelsize=axis_tick_size)

                axs[4, (ses-1)].axvline(x=1, lw=lw_value, clip_on=False,color=(134/256.0,250/256.0,167/256.0))
                axs[4, (ses-1)].axvline(x=0.9, lw=lw_value, clip_on=False,color=(128/256.0,171/256.0,69/256.0))
                axs[4, (ses-1)].axvline(x=0.8, lw=lw_value, clip_on=False,color=(128/256.0,126/256.0,38/256.0))
                axs[4, (ses-1)].axvline(x=0.7, lw=lw_value, clip_on=False,color=(246/256.0,192/256.0,66/256.0))
                axs[4, (ses-1)].axvline(x=0.6, lw=lw_value, clip_on=False,color=(192/256.0,98/256.0,43/256.0))

            if pl >13 and pl <= 15:
                if pl==14:
                    color_plot='red'
                if pl==15:
                    color_plot='blue'
                axs[5, (ses-1)].fill_between(x_grid, data_lb,data_up,color=color_plot)
                axs[5, (ses-1)].plot(x_grid, datamean,linewidth=0.3,color='black')
                axs[5,(ses-1)].tick_params(axis='both', which='major', labelsize=axis_tick_size)

                axs[5, (ses-1)].axvline(x=1, lw=lw_value, clip_on=False,color=(134/256.0,250/256.0,167/256.0))
                axs[5, (ses-1)].axvline(x=0.9, lw=lw_value, clip_on=False,color=(128/256.0,171/256.0,69/256.0))
                axs[5, (ses-1)].axvline(x=0.8, lw=lw_value, clip_on=False,color=(128/256.0,126/256.0,38/256.0))
                axs[5, (ses-1)].axvline(x=0.7, lw=lw_value, clip_on=False,color=(246/256.0,192/256.0,66/256.0))
                axs[5, (ses-1)].axvline(x=0.6, lw=lw_value, clip_on=False,color=(192/256.0,98/256.0,43/256.0))

    plt.savefig(datain + '/Mult_session_spatial_corr_test_new_combined_new.png')
