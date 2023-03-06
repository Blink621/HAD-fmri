import os
import numpy as np
import nibabel as nib
from os.path import join as pjoin
from had_utils import save2cifti
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

# define path
# make sure the dataset_path are modified based on your personal dataset downloading directory
dataset_path = '/nfs/z1/userhome/ZhouMing/workingdir/BIN/data_upload/HAD'
ciftify_path = f'{dataset_path}/derivatives/ciftify'
nifti_path = f'{dataset_path}'
support_path = './support_files'
# change to path of current file
os.chdir(os.path.dirname(__file__))

# prepare params
sub_names = ['sub-%02d'%(i+1) for i in range(30)]
# sub_names = ['sub-01']
n_cycle = 4
n_sub = len(sub_names)

# start computing reliability 
reliability_sum = np.zeros((n_sub, 59412))
for sub_idx, sub_name in enumerate(sub_names):
    # extract beta in each cycle results
    beta_sub = np.zeros((4, 180, 59412))
    for cycle_idx in range(n_cycle):
        beta_cycle = nib.load(pjoin(ciftify_path, sub_name, 'results', f'ses-action01_task-action_cycle-{cycle_idx+1}_beta.dscalar.nii')).get_fdata()
        # scale data
        scaler = StandardScaler()
        beta_sub[cycle_idx] = scaler.fit_transform(beta_cycle)
    # generate odd cycle and even cycle pattern
    odd_cycle_pattern = beta_sub[[0, 2], :, :].mean(axis=0)
    even_cycle_pattern = beta_sub[[1, 3], :, :].mean(axis=0)
    # iterate on voxel to generate reliability
    for voxel_idx in range(beta_sub.shape[-1]):
        reliability_sum[sub_idx, voxel_idx] = pearsonr(odd_cycle_pattern[:, voxel_idx], even_cycle_pattern[:, voxel_idx])[0]
    print(f'Finish computing reliability in {sub_name}')
# save reliability map
temp = nib.load(pjoin(support_path, 'template.dtseries.nii'))
reliability_map = np.zeros((91282))
reliability_map[:59412] = reliability_sum.mean(axis=0)
reliability_path = pjoin(support_path, f'reliability.dtseries.nii')
save2cifti(file_path=reliability_path, data=reliability_map, brain_models=temp)

