
import os
import glob
import subprocess
import numpy as np

"""
This file is to change file names on ciftify time series or nii file


"""
sub_id = np.arange(30)
for subj in sub_id:
    sub_name = 'sub-{:02d}'.format(subj+1)
    ses = 'ses-action*'
    run = 'run-*'

    old_files_log = '/nfs/z1/zhenlab/BrainImageNet/action/data/bold/derivatives/ciftify/{0}' \
                    '/MNINonLinear/Results/{1}_task-action_{2}/' \
                    'ciftify_subject_fmri.log'.format(sub_name, ses, run)
    old_files_dt = '/nfs/z1/zhenlab/BrainImageNet/action/data/bold/derivatives/ciftify/{0}' \
                   '/MNINonLinear/Results/{1}_task-action{2}/' \
                   '{1}_task-action_{2}_Atlas_s0.dtseries.nii'.format(sub_name, ses, run)
    old_files_fmriprep = '/nfs/z1/zhenlab/BrainImageNet/action/data/bold/derivatives/fmriprep/{0}/' \
                   '{1}/func/{0}_{1}_task-action_{2}_space-T1w_desc-preproc_bold_hp2000.nii.gz'.format(sub_name, ses, run)
    old_files_melodic = '/nfs/z1/zhenlab/BrainImageNet/action/data/bold/derivatives/melodic/{0}/' \
                   '{1}/{0}_{1}_task-action{2}.ica/remove_thresh-0.txt'.format(sub_name, ses, run)
    old_files_gz = '/nfs/z1/zhenlab/BrainImageNet/action/data/bold/derivatives/ciftify/{0}' \
                   '/MNINonLinear/Results/{1}_task-action_{2}/' \
                   '{1}_task-action_{2}_org.nii.gz'.format(sub_name, ses, run)                   

    old_files_log = sorted(glob.glob(old_files_log))
    old_files_dt = sorted(glob.glob(old_files_dt))
    old_files_fmriprep = sorted(glob.glob(old_files_fmriprep))
    old_files_melodic = sorted(glob.glob(old_files_melodic))
    old_files_gz = sorted(glob.glob(old_files_gz))

    for x in old_files_fmriprep:
        print(x) 
    # check = input('Please check if you want to delete: (y/N)')
    # if check == 'y':
    for old_file in old_files_fmriprep:
        cmd = ' '.join(['rm', old_file])
        print(cmd)
        try:
            subprocess.check_call(cmd, shell=True)
        except subprocess.CalledProcessError:
            raise Exception('mv: Error happened ')

        # for old_file in old_files_fmriprep:
        #     new_file = old_file.replace('Atlas_s0.dtseries.nii', 'Atlas_denoised_confound.dtseries.nii')
        #     cmd = ' '.join(['mv', old_file, new_file])
        #     print(cmd)
        #     try:
        #         subprocess.check_call(cmd, shell=True)
        #     except subprocess.CalledProcessError:
        #         raise Exception('mv: Error happened at {old_file}')

        # for old_file in old_files_gz:
        #     # if old_file.split('.')[0][-1].isdigit():
        #     new_file = old_file.replace('_org.nii.gz', '.nii.gz')
        #     cmd = ' '.join(['mv', old_file, new_file])
        #     print(cmd)
        #     try:
        #         subprocess.check_call(cmd, shell=True)
        #     except subprocess.CalledProcessError:
        #         raise Exception('mv: Error happened at {old_file}')
