# start docker service

sudo service docker start


# pull the cpac 1.8
docker pull fcpindi/c-pac:release-v1.8.0


# download the cpac:defautl no nuisance config

curl -L https://raw.githubusercontent.com/hahaai/Reproducibility_Analysis/master/AWS_Run/2021_Final/configs/cpac_1.8_default_no_nuisance.yml --output /home/ec2-user/configs/cpac_1.8_default_no_nuisance.yml


##################### CPAC:Default with 2mm write-out no nuisance #####################
## anat write-out res: 1mm, func write-out res: 2mm

output='/home/ec2-user/CPAC_1.8/CPAC_default_no_nuisance'
mkdir -p $output
chmod -R 777 $output
docker run \
-v /home/ec2-user/data_BIDS_All_sessions:/bids_dataset \
-v /$output:/outputs \
-v /tmp:/scratch -v /home/ec2-user/configs:/configs \
fcpindi/c-pac:release-v1.8.0 /bids_dataset /outputs participant \
--pipeline_file /configs/cpac_1.8_default_no_nuisance.yml \
--save_working_dir --mem_gb 14 --n_cpus 63 --pipeline_override "pipeline_setup: {system_config: {max_cores_per_participant: 2, maximum_memory_per_participant: 14, num_participants_at_once: 30, num_ants_threads: 2}}"




############ upload data to s3
while true;do
	aws s3 sync /home/ec2-user/CPAC_1.8/CPAC_default_no_nuisance s3://fcp-indi/data/uploads/CPAC_HNU/FINAL_preprocessed_2021/CPAC_default_no_nuisance --acl public-read
    echo 'sleep 5min'
    sleep 300
done


# check how many finished:
ls -l CPAC_1.8/CPAC_default_no_nuisance/output/cpac_cpac-default-pipeline/sub-00254*_ses-1/func/sub-00254*_ses-1_task-rest_space-template_desc-brain_bold.nii.gz | wc -l

