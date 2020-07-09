#Pipelines

##pipeline_config_Default.yml:

Defualt: 24 motion + CSF + GSR + ACompCor + Filter (0.01-0.1) + PlyOrt (2)

This is from the CPAD:Default, but changed the functional write out resoluiton to 2mm from 3mm.


##pipeline_config_FX_24-options.yml
	
Adapoted from here https://github.com/FCP-INDI/C-PAC/blob/66c5695b2e8dae46361d2ffb532fb52144f393a5/CPAC/resources/configs/pipeline_config_fmriprep-options.yml

Changes:
* turned on nuisanse regressor. 36 parameter with polynomial at 2.
* - collapse-output-transforms: 0
* func write out resolution to 2mm (matches the XCP write out) and changed reference to :/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain.nii.gz

