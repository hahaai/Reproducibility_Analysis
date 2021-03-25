import sys
#sys.path.append("/Users/leiai/projects/CPAC_fmriprep/Scripts")
from spatial_corr_ridgeplot_func import spatial_corr_ridgeplot
from spatial_corr_ridgeplot_func import ICC_ridgeplot




#pipelines=['cpac_fmriprep','abcd','fmriprep_1','fmriprep_MNI2004_2mm','dpabi','ccs']
#pipelines=['cpac_default_off_nuisance','abcd','fmriprep_1','fmriprep_MNI2004_2mm','dpabi','ccs']

#pipelines=['cpac_fmriprep','cpac_default_off_nuisance','fmriprep_1','fmriprep_MNI2004_2mm','ccs']

# The main redgeplot figure for the minimal preprocessing before matching parameters
pipelines=['cpac_default_off_nuisance','abcd','fmriprep_1','fmriprep_MNI2004_2mm','dpabi','ccs']


# summarry figure - minimal preprocessing
#pipelines=['fmriprep_1','cpac_fmriprep','cpac_default_off_nuisance']



# The main ridgeplot figure after matching parameters
pipelines=['cpac_fmriprep','cpac_default_off_nuisance','fmriprep_1','fmriprep_MNI2004_2mm','ccs']





## test
#pipelines=['fmriprep_1','XCP_on_fmriprep_nothing']


fc_handle=''
corr_type='pearson'
simpleplot=False
mulitple_atlas=True

if mulitple_atlas:
    atlases=['200','600','1000']

else:
    atlases=['200',]

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
            'fmriprep_MNI2009_2mm':'fMRIPrep',
            'fmriprep_1':'fMRIPrep',
            'fmriprep_MNI2004_2mm':'fMRIPrep',
            'fmriprep_MNI2006_native':'fMRIPrep'
            }

base='/Users/leiai/projects/CPAC_fmriprep/Reliability_discriminibility/Finalizing/Minimal/ROI'



######## Plot
'''
# The main redgeplot figure for the minimal preprocessing before matching parameters

# fmriprep default
pipelines=['cpac_default_off_nuisance','abcd','fmriprep_1','dpabi','ccs']
spatial_corr_ridgeplot(base,base.replace('ROI','figures'),pipelines,atlases,namechangedict,fc_handle,simpleplot,'pearson')

# fmriprep MNI2006 2mm
pipelines=['cpac_default_off_nuisance','abcd','fmriprep_MNI2004_2mm','dpabi','ccs']
spatial_corr_ridgeplot(base,base.replace('ROI','figures'),pipelines,atlases,namechangedict,fc_handle,simpleplot,'pearson')

# fmriprep MNI2009 2mm
pipelines=['cpac_default_off_nuisance','abcd','fmriprep_MNI2009_2mm','dpabi','ccs']
spatial_corr_ridgeplot(base,base.replace('ROI','figures'),pipelines,atlases,namechangedict,fc_handle,simpleplot,'pearson')
'''
# fmriprep MNI2006 3.4mm - don't hoave now
#pipelines=['cpac_default_off_nuisance','abcd','fmriprep_MNI2006_native','dpabi','ccs']
#spatial_corr_ridgeplot(base,base.replace('ROI','figures'),pipelines,atlases,namechangedict,fc_handle,simpleplot,'pearson')




####### The main ridgeplot figure after matching parameters
'''
# fmriprep default
pipelines=['cpac_fmriprep','cpac_default_off_nuisance','fmriprep_1','ccs']
spatial_corr_ridgeplot(base,base.replace('ROI','figures'),pipelines,atlases,namechangedict,fc_handle,simpleplot,'pearson')

# fmriprep mni2004 with 2mm
pipelines=['cpac_fmriprep','cpac_default_off_nuisance','fmriprep_MNI2004_2mm','ccs']
spatial_corr_ridgeplot(base,base.replace('ROI','figures'),pipelines,atlases,namechangedict,fc_handle,simpleplot,'pearson')

# fmriprep mni2009 with 2mm
pipelines=['cpac_fmriprep','cpac_default_off_nuisance','fmriprep_MNI2009_2mm','ccs']
spatial_corr_ridgeplot(base,base.replace('ROI','figures'),pipelines,atlases,namechangedict,fc_handle,simpleplot,'pearson')
'''

# fmriprep mni2006 3.4 mm don't have
#pipelines=['cpac_fmriprep','cpac_default_off_nuisance','fmriprep_MNI2006_native','ccs']
#spatial_corr_ridgeplot(base,base.replace('ROI','figures'),pipelines,atlases,namechangedict,fc_handle,simpleplot,'pearson')





# spatial_corr_ridgeplot(base,base.replace('ROI','figures'),pipelines,atlases,namechangedict,fc_handle,simpleplot,'pearson')
# spatial_corr_ridgeplot(base,base.replace('ROI','figures'),pipelines,atlases,namechangedict,fc_handle,simpleplot,'spearman')
# spatial_corr_ridgeplot(base,base.replace('ROI','figures'),pipelines,atlases,namechangedict,fc_handle,simpleplot,'concordance')



########################################################################### Final Figures:

# Figure1.a - Minimal preprocessing (Before harmonization)  

# Minimal preprocessing (Before harmonization) 
# panel 1:
# Different template/Different write-out due to default settings -fMRIPrep in MNI2009 and native func resolution
# Ridgeplot_spatial_corr_pearson_cpac_default_off_nuisance-abcd-fmriprep_1-dpabi-ccs_200.png
# Ridgeplot_spatial_corr_pearson_cpac_default_off_nuisance-abcd-fmriprep_1-dpabi-ccs_1000.png
pipelines=['cpac_default_off_nuisance','abcd','fmriprep_2','dpabi','ccs']
nameorder=spatial_corr_ridgeplot(base,base.replace('ROI','figures'),pipelines,atlases,namechangedict,fc_handle,simpleplot,'pearson')
#nameorder=''
ICC_ridgeplot(base.replace('ROI','figures'),base.replace('ROI','figures'),pipelines,atlases,namechangedict,simpleplot,nameorder)

with open(base.replace('ROI','figures') + '/Ridgeplot_'+'-'.join(pipelines)+'_PlotNameOrder.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % place for place in nameorder)


# Figure1.b - Minimal preprocessing (Before harmonization)  
# Minimal preprocessing (Before harmonization)  
# - Same template/Same write-out fMRIPrep in MNI2006 and 2mm resolution
# Ridgeplot_spatial_corr_pearson_cpac_default_off_nuisance-abcd-fmriprep_MNI2004_2mm-dpabi-ccs_200.png
# Ridgeplot_spatial_corr_pearson_cpac_default_off_nuisance-abcd-fmriprep_MNI2004_2mm-dpabi-ccs_1000.png
pipelines=['cpac_default_off_nuisance','abcd','fmriprep_MNI2004_2mm','dpabi','ccs']
nameorder=spatial_corr_ridgeplot(base,base.replace('ROI','figures'),pipelines,atlases,namechangedict,fc_handle,simpleplot,'pearson')
#nameorder=''
ICC_ridgeplot(base.replace('ROI','figures'),base.replace('ROI','figures'),pipelines,atlases,namechangedict,simpleplot,nameorder)

with open(base.replace('ROI','figures') + '/Ridgeplot_'+'-'.join(pipelines)+'_PlotNameOrder.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % place for place in nameorder)

'''
# Figure2 - Minimal preprocessing (After harmonization) - Same template/Same write-out  
#Same template/Same write-out  
#fMRIPrep in MNI2006 and 2mm resolution
# Ridgeplot_spatial_corr_pearson_cpac_fmriprep-cpac_default_off_nuisance-fmriprep_1-ccs_200.png
# Ridgeplot_spatial_corr_pearson_cpac_fmriprep-cpac_default_off_nuisance-fmriprep_1-ccs_1000.png
pipelines=['cpac_fmriprep','cpac_default_off_nuisance','fmriprep_2','ccs']
nameorder=spatial_corr_ridgeplot(base,base.replace('ROI','figures'),pipelines,atlases,namechangedict,fc_handle,simpleplot,'pearson')
ICC_ridgeplot(base.replace('ROI','figures'),base.replace('ROI','figures'),pipelines,atlases,namechangedict,simpleplot,nameorder)

with open(base.replace('ROI','figures') + '/Ridgeplot_'+'-'.join(pipelines)+'_PlotNameOrder.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % place for place in nameorder)
'''

# Figure2, with CPAC:fMRIPrep replace CPAC:Default

# - Same template/Same write-out fMRIPrep in MNI2006 and 2mm resolution
# Ridgeplot_spatial_corr_pearson_cpac_default_off_nuisance-abcd-fmriprep_MNI2004_2mm-dpabi-ccs_200.png
# Ridgeplot_spatial_corr_pearson_cpac_default_off_nuisance-abcd-fmriprep_MNI2004_2mm-dpabi-ccs_1000.png
#pipelines=['cpac_fmriprep_MNI2004_2mm','abcd','fmriprep_MNI2004_2mm','dpabi','ccs']
#pipelines=['cpac_fmriprep_MNI2004_2mm','fmriprep_MNI2004_2mm','cpac_default_off_nuisance','fmriprep_2']

pipelines=['cpac_fmriprep_MNI2004_2mm','fmriprep_MNI2004_2mm','cpac_default_off_nuisance']
nameorder=spatial_corr_ridgeplot(base,base.replace('ROI','figures'),pipelines,atlases,namechangedict,fc_handle,simpleplot,'pearson')
#nameorder=''
ICC_ridgeplot(base.replace('ROI','figures'),base.replace('ROI','figures'),pipelines,atlases,namechangedict,simpleplot,nameorder)

with open(base.replace('ROI','figures') + '/Ridgeplot_'+'-'.join(pipelines)+'_PlotNameOrder.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % place for place in nameorder)



# Figure 2.1 - Minimal preprocessing (After harmonization) - Same template/Same write-out    - MNI2004 2mm for both fMRIPrep and CPAC:fMRIPrep
#Same template/Same write-out  
#fMRIPrep in MNI2006 and 2mm resolution
# Ridgeplot_spatial_corr_pearson_cpac_fmriprep-cpac_default_off_nuisance-fmriprep_1-ccs_200.png
# Ridgeplot_spatial_corr_pearson_cpac_fmriprep-cpac_default_off_nuisance-fmriprep_1-ccs_1000.png
pipelines=['cpac_fmriprep_MNI2004_2mm','cpac_default_off_nuisance','fmriprep_MNI2004_2mm','ccs']
nameorder=spatial_corr_ridgeplot(base,base.replace('ROI','figures'),pipelines,atlases,namechangedict,fc_handle,simpleplot,'pearson')
ICC_ridgeplot(base.replace('ROI','figures'),base.replace('ROI','figures'),pipelines,atlases,namechangedict,simpleplot,nameorder)

with open(base.replace('ROI','figures') + '/Ridgeplot_'+'-'.join(pipelines)+'_PlotNameOrder.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % place for place in nameorder)


# Figure 2.1.B - Minimal preprocessing (After harmonization) - Same template/Same write-out    - MNI2004 2mm for both fMRIPrep and CPAC:fMRIPrep
# With CPAC:ABCD
#Same template/Same write-out  
#fMRIPrep in MNI2006 and 2mm resolution
# Ridgeplot_spatial_corr_pearson_cpac_fmriprep-cpac_default_off_nuisance-fmriprep_1-ccs_200.png
# Ridgeplot_spatial_corr_pearson_cpac_fmriprep-cpac_default_off_nuisance-fmriprep_1-ccs_1000.png
pipelines=['CPAC_ABCD_scratch_0309_final','cpac_default_off_nuisance','abcd']
nameorder=spatial_corr_ridgeplot(base,base.replace('ROI','figures'),pipelines,atlases,namechangedict,fc_handle,simpleplot,'pearson')
ICC_ridgeplot(base.replace('ROI','figures'),base.replace('ROI','figures'),pipelines,atlases,namechangedict,simpleplot,nameorder)


'''
ICC_ridgeplot(base.replace('ROI','figures'),base.replace('ROI','figures'),pipelines,atlases,namechangedict,simpleplot,nameorder)

with open(base.replace('ROI','figures') + '/Ridgeplot_'+'-'.join(pipelines)+'_PlotNameOrder.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % place for place in nameorder)
'''


# Figure break down steps:

# Steps breaking down, all with MNI2004-2mm
simpleplot=True
atlases=['200',]
pipelines=['fmriprep_MNI2004_2mm','cpac_fmriprep_MNI2004_2mm','cpac_default_off_nuisance','cpac_fmriprep_MNI2004_2mm_anat_mask','cpac_fmriprep_MNI2004_2mm_func_mask','cpac_fmriprep_MNI2004_2mm_coreg','cpac_fmriprep_MNI2004_2mm_N4','cpac_fmriprep_MNI2004_2mm_no_stc']
pipelines=['fmriprep_2','cpac_fmriprep','cpac_default_off_nuisance','cpac_fmriprep_anat_mask','cpac_fmriprep_func_mask','cpac_fmriprep_coreg','cpac_fmriprep_N4','cpac_fmriprep_stc']

nameorder=spatial_corr_ridgeplot(base,base.replace('ROI','figures'),pipelines,atlases,namechangedict,fc_handle,simpleplot,'pearson')
ICC_ridgeplot(base.replace('ROI','figures'),base.replace('ROI','figures'),pipelines,atlases,namechangedict,simpleplot,nameorder)

outfile=base.replace('ROI','figures') + '/Ridgeplot_'+'-'.join(pipelines)+'_PlotNameOrder.txt'
if len(os.path.basename(outfile))>140:
    outfile=os.path.dirname(outfile)+ '/' + os.path.basename(outfile)[-140:]

with open(outfile, 'w') as filehandle:
    filehandle.writelines("%s\n" % place for place in nameorder)



'''
#### testing:
simpleplot=True
atlases=['200',]
pipelines=['abcd','cpac_default_off_nuisance','CPAC_default_NLM','CPAC_default_reg_without_skull']
nameorder=spatial_corr_ridgeplot(base,base.replace('ROI','figures'),pipelines,atlases,namechangedict,fc_handle,simpleplot,'pearson')
'''
