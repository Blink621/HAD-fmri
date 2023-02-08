import os
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.linear_model import Ridge
from os.path import join as pjoin
from fracridge import FracRidgeRegressor
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix

def save_ciftifile(data, filename):
    template = '/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/Analysis_derivatives/ciftify/sub-core02/MNINonLinear/Results/ses-ImageNet01_task-object_run-1/ses-ImageNet01_task-object_run-1_Atlas.dtseries.nii'
    ex_cii = nib.load(template)
    if len(data.shape) > 1:
        ex_cii.header.get_index_map(0).number_of_series_points = data.shape[0]
    else:
        ex_cii.header.get_index_map(0).number_of_series_points = 1
        data = data[np.newaxis, :]
    nib.save(nib.Cifti2Image(data.astype(np.float32), ex_cii.header), filename)

# define path
beta_path = '/nfs/z1/zhenlab/BrainImageNet/action/data/bold/derivatives/beta'
ciftify_path = '/nfs/z1/zhenlab/BrainImageNet/action/data/bold/derivatives/ciftify'
nifti_path = '/nfs/z1/zhenlab/BrainImageNet/action/data/bold/nifti'
result_path = '/nfs/z1/zhenlab/BrainImageNet/action/data_paper/result'

# prepare params
sub_names = sorted([i for i in os.listdir(beta_path) if i.startswith('sub')])
# sub_names = ['sub-01']
alpha = 0.1

for sub_idx, sub_name in enumerate(sub_names):
    sub_tmp_path = pjoin(nifti_path, sub_name)
    # if not os.path.exists(beta_clean_path):
    # prepare params
    tr, begin_dur, n_tr, n_event, n_run, n_class = 2, 12, 156, 60, 12, 180
    frame_times = np.arange(n_tr*3) * tr 
    # define beta path
    sub_tmp_path = pjoin(nifti_path, sub_name)
    _result_path = pjoin(ciftify_path, sub_name, 'MNINonLinear/Results/')
    sess_name = 'ses-action01'
    # loop in one subject
    sub_func_path = pjoin(sub_tmp_path, sess_name, 'func')
    events_file = sorted([i for i in os.listdir(sub_func_path) if 'events' in i and 'rest' not in i and \
                        int(i.split('-')[-1].split('_')[0])<=12 and 'discard' not in i])

    for run_idx in range(12):
        # prepare basic path
        beta_clean_path = pjoin(_result_path, run_name, f'{run_name}_beta.dtseries.nii')
        events_file_name = '%s_ses-action01_task-action_run-%02d_events.tsv'%(sub_name, run_idx + 1)
        run_name = 'ses-action01_task-action_run-%d'% (run_idx + 1)
        # fit design matrix based on trial onset time
        events_raw = pd.read_csv(pjoin(sub_func_path, events_file_name), sep='\t')
        duration = events_raw['duration']
        onset = events_raw['onset'].to_numpy() + begin_dur
        label_tmp = events_raw['trial_type'].to_numpy()
        trial_type = ['image%03d'%idx for idx in label_tmp]
        # load time series
        dtseries_path = pjoin(_result_path, run_name, f'{run_name}_Atlas.dtseries.nii')
        dtseries = nib.load(dtseries_path).get_fdata()
        print(f'load {dtseries_path}')
    # prepare design matrix
    events = pd.DataFrame({'trial_type':trial_type, 'onset':onset, 'duration':duration})
    design_matrix = make_first_level_design_matrix(frame_times, events, drift_order=2, hrf_model='spm')
    design_matrix.drop(design_matrix.columns[-1], axis=1, inplace=True)
    # perform GLM
    reg = Ridge(alpha=alpha, fit_intercept=False).fit(design_matrix.values, dtseries)
    beta_clean = reg.coef_[:, :n_class].transpose(1,0).astype(np.float32)
    print('Finish performing GLM in %s %s run %02d'%(sub_name, sess_name, run_idx+1))
    # save data
    save_ciftifile(beta_clean, beta_clean_path)
