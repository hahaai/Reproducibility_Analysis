# start docker service

sudo service docker start


# pull the fMRIPrep LTS
docker pull nipreps/fmriprep:20.2.1



##################### Default settings mni2009 + native func write out #####################
output='/home/ec2-user/fmriprep/output'
working='/home/ec2-user/fmriprep/working'
mkdir -p $output
mkdir -p $working
chmod -R 777 $output
chmod -R 777 $working
docker run --rm -i \
-v /home/ec2-user/configs:/freesurfer \
-v /home/ec2-user/data_BIDS:/bids_dataset  \
-v $output:/out \
-v $working:/scratch nipreps/fmriprep:20.2.1 /bids_dataset /out participant \
--skip_bids_validation --fs-license-file /freesurfer/license.txt \
-w /scratch --n_cpus 63 --nthreads 63  --fs-no-reconall


############ upload data to s3
while true;do
	aws s3 sync /home/ec2-user/fmriprep s3://fcp-indi/data/uploads/CPAC_HNU/FINAL_preprocessed_2021/fmriprep_default --acl public-read
    echo 'sleep 5min'
    sleep 300
done

## download to ned
while true;do 
	aws s3 sync s3://fcp-indi/data/uploads/CPAC_HNU/FINAL_preprocessed_2021/fmriprep_default /data3/cnl/fmriprep/Lei_working/FINAL_preprocessed_2021/fmriprep_default 
	#--exclude "*" --include "*vol0000_warp_merged_maths.nii.gz*"
	echo 'sleep 5min'  
	sleep 60
done
