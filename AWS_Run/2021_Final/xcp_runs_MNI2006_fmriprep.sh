
###### xcp runs 

# setup


# here change the design file accourdingly
dsfilename='design_cpac.dn'
dsfilename='design_fmriprep.dn'

outfix=''
#dsfilename='design_cpac_reg.dn'
#outfix='_reg'


#dsfilename='design_cpac_24.dn'
#outfix=''

#dsfilename='design_cpac_24_reg.dn'
#outfix='_reg'

commandlist='/home/ec2-user/xcp_commands_all_fmriprep.txt'
rm $commandlist


#for xx in nuisance filter reg_filter;do
#for xx in nuisance filter reg_filter;do
for xx in reg_filter;do

#for xx in reg_filter;do
dsfilename='design_fmriprep_'$xx'.dn'
outfix='_'$xx

if [[ $xx == 'reg_filter' ]];then
    xx=''
   dsfilename='design_fmriprep.dn'
    outfix=''
fi


for run in fmriprep;do
    #run=cpac_FX
    # run=CPAC_FX_2mm
    #for tmp in $(ls 'CPAC_1.6.2_xcprun/'$run'/output');do
    for tmp in $(ls 'CPAC_1.6.2/'$run'/output');do
        echo $tmp
    done

    datain='/home/ec2-user/fmriprep/output/fmriprep'
    datain='/home/ec2-user/fmriprep/output/fmriprep'

    ds_folder='/home/ec2-user/designfile'

    mkdir -p /home/ec2-user/cohort

    x='/home/ec2-user'

    basefolder='/home/ec2-user'

    bind=${HOME}'/data_tmp/'$run
    mkdir -p $bind

    mkdir -p '/home/ec2-user/cohort'
    mkdir -p '/home/ec2-user/xcpout'

    rm -r '/home/ec2-user/cohort/'$run
    # /data3/cnl/fmriprep/AllNewRun/fmriprep_2/output/fmriprep/sub-0025427a/func/sub-0025427a_task-rest_run-1_space-T1w_desc-preproc_bold.nii.gz

    for sub in $(ls $datain);do

        if [[ $sub == *"html"* ]];then continue;fi
        if [[ $sub != *"sub-"* ]];then continue;fi

        echo $sub
        sub_short=${sub:0:12}

        file=[]
        #for file in $(find $datain'/'$sub'/ses-1/func/' -iname '*task-rest_space-MNI1522004_res-2_desc-preproc_bold.nii.gz*');do
        for file in $(find $datain'/'$sub'/func/' -iname '*task-rest_run-1_space-MNI1522004_desc-preproc_bold.nii.gz*');do
            echo $file
        done

        if [[ -f $file ]] && [[ ! -f 'xcpout/'$run'/'$sub_short'/norm/'$sub_short'_std.nii.gz' ]];then
            echo $file
            cohort=$basefolder'/cohort/'$run'/'$sub_short.csv
            cohort_in=${cohort/$x/$bind}
            mkdir -p $(dirname $cohort)
            echo id0,img >> $cohort
            echo $sub_short,${file/$x/$bind} >> $cohort
            echo singularity run --cleanenv -B /home/ec2-user:$bind /home/ec2-user/xcpEngine_MNI2006.simg -c $cohort_in -d $basefolder'/configs/xcp_designfile/'$dsfilename -o $bind'/xcpout/'$run$outfix -r $bind >> $commandlist
        fi
    done
done


done

# singularity run --cleanenv -B /home/ec2-user:/home/ec2-user/data/cpac_FX /home/ec2-user/xcpEngine.simg -c /home/ec2-user/data/cpac_FX/cohort/cpac_FX/sub-0025427a.csv -d /home/ec2-user/data/cpac_FX/designfile/design_cpac.dn -o /home/ec2-user/data/cpac_FX/xcpout/cpac_FX -r /home/ec2-user/data/cpac_FX


