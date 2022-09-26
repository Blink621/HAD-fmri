import os
import subprocess
import numpy as np
from os.path import join as pjoin
from numpy import dtype
from tqdm import tqdm

# data path
proj_dir = '/nfs/z1/zhenlab/BrainImageNet/action'
fmriprep_dir = 'data/bold/derivatives/fmriprep'
fmriprep_path = pjoin(proj_dir, fmriprep_dir)

# preare commands
os.chdir(fmriprep_path)
func_dir_temp = 'sub-%02d/ses-action%02d/func/'
file_name_temp = 'sub-%02d_ses-action%02d_task-action_run-%d_space-T1w_desc-preproc_bold.nii.gz'
subjects = np.linspace(20,29,10,dtype=int)
sessions = [1]
runs = range(12)
highpass, tr = 128, 2
cmds, exp_func_files = [], []
func_files = []
for sub in subjects:
    for ses in sessions:
        func_dir = func_dir_temp % (sub+1, ses)
        func_files.extend([_ for _ in os.listdir(pjoin(fmriprep_path, func_dir)) if 'desc-preproc_bold.nii' in _ and 'action' in _])
        for run in runs:
            fmri_file = file_name_temp % (sub+1, ses, run+1)
            fmri = pjoin(fmriprep_path, func_dir, fmri_file)
            fmri_hp = fmri.replace('desc-preproc_bold', 'desc-preproc_bold_hp128')
            hptr = '%.5f' % (highpass/(2*tr))
            # cmd
            demean_cmd = 'fslmaths {0} -Tmean {1}'.format(fmri, fmri_hp)
            highpass_cmd = "fslmaths {0} -sub {1} -bptf {2} -1 -add {1} {1}".format(fmri, fmri_hp, hptr)
            # store cmd
            cmds.append(demean_cmd)
            cmds.append(highpass_cmd)
            exp_func_files.append(fmri_file)


# # run commands
# for cmd in tqdm(cmds):
#     print(cmd)
#     res = subprocess.check_call(cmd,shell=True)

# # for remainning files
# cmds=[]
# remain_func_files = list(set(func_files) - set(exp_func_files))
# for file in remain_func_files:
#     sub, ses = int(file.split('_')[0][-2::]), int(file.split('_')[1][-2::]) 
#     func_dir = func_dir_temp % (sub, ses)
#     fmri = pjoin(fmriprep_path, func_dir, file)
#     fmri_hp = fmri.replace('desc-preproc_bold', 'desc-preproc_bold_hp2000')
#     hptr = '%.5f' % (highpass/(2*tr))
#     # cmd
#     demean_cmd = 'fslmaths {0} -Tmean {1}'.format(fmri, fmri_hp)
#     highpass_cmd = "fslmaths {0} -sub {1} -bptf {2} -1 -add {1} {1}".format(fmri, fmri_hp, hptr)
#     # store cmd
#     cmds.append(demean_cmd)
#     cmds.append(highpass_cmd)

# run commands
for cmd in tqdm(cmds):
    print(cmd)
    res = subprocess.check_call(cmd,shell=True)

#####
# checking codes
#####
# os.chdir('/nfs/z1/userhome/GongZhengXin/workingdir/fMRI/')
# for cmd in cmds[0:2]:
#     cmd = cmd.replace('sub-01/ses-ImageNet01/func/', '')
#     print(cmd)
#     res = subprocess.check_call(cmd,shell=True)

# import nibabel as nib
# bold_raw = nib.load('sub-01_ses-ImageNet01_task-naturalvision_run-1_space-T1w_desc-preproc_bold.nii.gz')
# bold_hp = nib.load('sub-01_ses-ImageNet01_task-naturalvision_run-1_space-T1w_desc-preproc_bold_hp2000.nii.gz')