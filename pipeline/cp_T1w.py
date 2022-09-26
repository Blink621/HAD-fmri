import os
import subprocess
from os.path import join as pjoin

imagenet_path = '/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/nifti'
action_path = '/nfs/z1/zhenlab/BrainImageNet/action/data/bold/nifti'

# the key reflect the action subID, the content reflect the imagenet subID
correspond = {'sub-01':'sub-12', 'sub-02':'sub-20', 'sub-03':'sub-14', 'sub-04':'sub-25', 'sub-05':'sub-17',
              'sub-06':'sub-08', 'sub-07':'sub-33', 'sub-08':'sub-16', 'sub-09':'sub-13', 'sub-10':'sub-21',
              'sub-12':'sub-22', 'sub-13':'sub-05', 'sub-15':'sub-31', 'sub-19':'sub-28', 'sub-20':'sub-10',
              }


for sub_action_name in correspond.keys():
    sub_imagenet_name = correspond[sub_action_name]
    sub_imagenet_path = pjoin(imagenet_path, sub_imagenet_name)
    sub_action_path = pjoin(action_path, sub_action_name)
    # find anat folder in imagenet nifti folder
    for sess in os.listdir(sub_imagenet_path):
        if 'anat' in os.listdir(pjoin(sub_imagenet_path, sess)):
            anat_imagenet_path = pjoin(sub_imagenet_path, sess, 'anat')
            anat_action_path = pjoin(sub_action_path, 'ses-action01')
            # if not os.path.exists(anat_action_path):
            #     os.makedirs(anat_action_path)
            # Start copying folders
            cmd = ' '.join(['cp -r', anat_imagenet_path, anat_action_path])
            try:
                subprocess.check_call(cmd, shell=True)
            except subprocess.CalledProcessError:
                raise Exception(f'cp: Error happened at {sub_action_name}')
            # Rename the anat file
            old_files_gz = '{0}/anat/{1}_{2}_run-01_T1w.nii.gz'.format(anat_action_path, sub_imagenet_name, sess)
            new_files_gz = '{0}/anat/{1}_ses-action01_run-01_T1w.nii.gz'.format(anat_action_path, sub_action_name)
            old_files_json = '{0}/anat/{1}_{2}_run-01_T1w.json'.format(anat_action_path, sub_imagenet_name, sess)
            new_files_json = '{0}/anat/{1}_ses-action01_run-01_T1w.json'.format(anat_action_path, sub_action_name)
            # call cmd
            cmd_gz = ' '.join(['mv -v', old_files_gz, new_files_gz])
            cmd_json = ' '.join(['mv -v', old_files_json, new_files_json])
            try:
                subprocess.check_call(cmd_gz, shell=True)
                subprocess.check_call(cmd_json, shell=True)
            except subprocess.CalledProcessError:
                raise Exception(f'mv: Error happened at {sub_action_name}')



