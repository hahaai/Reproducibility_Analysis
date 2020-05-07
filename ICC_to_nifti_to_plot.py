print __doc__

from surfer import Brain, io
import glob

import numpy as np
import pandas as pd
import nibabel as nib
import os
import sys


atlas_file='/Users/leiai/projects/CPAC_fmriprep/Reliability_discriminibility/Finalizing/Schaefer_Atlas/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz'
datain='/Users/leiai/projects/CPAC_fmriprep/Reliability_discriminibility/Finalizing_HBN/Minimal/figures/ICC_Schaefer200'
num_roi=200




def vec2symMat(vect,n):
    mat=np.zeros(shape=(n,n))
    pre=0
    for i in range(0,n):
        mat[i,i]=1
        if i != n:
            mat[i,(i+1):n] = vect[pre:(pre+n-i-1)]
            mat[(i+1):n,i] = vect[pre:(pre+n-i-1)]
            pre=(pre+n-i-1)
            #print(pre)
    return mat

# laod atlas file
img = nib.load(atlas_file)
atlas=img.get_fdata()

for file in os.listdir(datain):
    if 'ICC.csv' in file:
        print(file)
        csv=datain + '/' + file
        data=np.genfromtxt(csv)
        m=vec2symMat(data,num_roi)
        fc_avg=(np.sum(m,axis=0)-1)/num_roi
        tmp=atlas.copy()
        for i in range(0,num_roi):
            tmp[tmp==(i+1)]=fc_avg[i]
    
        clipped_img = nib.Nifti1Image(tmp, img.affine, img.header)
        nib.save(clipped_img, datain + '/' + file.replace('.csv','.nii.gz'))


# put negative and positve togher.

os.environ["FREESURFER_HOME"] = "/Applications/freesurfer"
#os.environ["SUBJECTS_DIR"] = "/Users/lei.ai/Surface_display"
os.environ["SUBJECTS_DIR"]="/Applications/freesurfer/subjects"

#
#for mri_file in glob.glob("/Users/lei.ai/Documents/OneDrive\ -\ Child\ Mind\ Institute/projects/Peers/feat/data/*"):
#for mri_file in glob.glob("/Users/lei.ai/Documents/projects/CPAC_fmriprep/Reliability_discriminibility/Finalizing/Minimal/figures/ICC_Schaefer1000/*.nii.gz"):
for mri_file in glob.glob(datain + "/*.nii.gz"):

    #mri_file='/Users/lei.ai/Documents/projects/Peers/feat/data/DM_Pos.nii.gz'
    #save_name=mri_file.replace("data","Figures")
    save_name=mri_file.replace("nii.gz","png")
    
    if not os.path.isfile(save_name):
    	if not os.path.exists(os.path.dirname(save_name)):
    		os.makedirs(os.path.dirname(save_name))

        """Bring up the visualization"""
        brain = Brain("fsaverage", "split", "inflated", views=['lat', 'med'],
                      config_opts=dict(background="white"))

        """Project the volume file and return as an array"""
        reg_file = "/Applications/freesurfer/average/mni152.register.dat"
        surf_data_lh = io.project_volume_data(mri_file, "lh", reg_file)
        surf_data_rh = io.project_volume_data(mri_file, "rh", reg_file)
        #print(surf_data_lh)

        """
        You can pass this array to the add_overlay method for
        a typical activation overlay (with thresholding, etc.)
        """
        #brain.add_overlay(surf_data_lh, min=0.5, max=1,name="ang_corr_lh", hemi='lh')
        brain.add_data(surf_data_rh, min=0.01, max=1, hemi='rh',,colormap='jet')
        brain.add_data(surf_data_lh, min=0.01, max=1, hemi='lh',colormap='jet')
        brain.save_image(save_name)
    """
    You can also pass it to add_data for more control
    over the visualzation. Here we'll plot the whole
    range of correlations
    """
    #for overlay in brain.overlays_dict["ang_corr_lh"]:
    #    overlay.remove()
    #for overlay in brain.overlays_dict["ang_corr_rh"]:
     #   overlay.remove()

    #brain.add_data(surf_data_lh, 1, 4.2, colormap="jet", alpha=0.75, hemi='lh',colorbar="FALSE")
    #brain.add_data(surf_data_rh, 1, 4.2, colormap="jet", alpha=0.75, hemi='rh',colorbar=FALSE)

"""
This overlay represents resting-state correlations with a
seed in left angular gyrus. Let's plot that seed.
"""
#seed_coords = (-45, -67, 36)
#brain.add_foci(seed_coords, map_surface="white", hemi='lh')
