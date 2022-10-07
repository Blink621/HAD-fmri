import os
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.linear_model import Ridge
from os.path import join as pjoin
from fracridge import FracRidgeRegressor
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix

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
    # prepare basic path
    beta_clean_path = pjoin(beta_path, sub_name, f'{sub_name}_action-beta_clean_split.npy')
    sub_tmp_path = pjoin(nifti_path, sub_name)
    # if not os.path.exists(beta_clean_path):
    # prepare params
    tr, begin_dur, n_tr, n_event, n_run, n_class = 2, 12, 156, 60, 12, 180
    frame_times = np.arange(n_tr*3) * tr 
    # define beta path
    sub_tmp_path = pjoin(nifti_path, sub_name)
    _result_path = pjoin(ciftify_path, sub_name, 'MNINonLinear/Results/')
    sess_name = 'ses-action01'
    beta_clean = np.zeros((4, n_class, 59412), dtype=np.float32)
    # loop in one subject
    sub_func_path = pjoin(sub_tmp_path, sess_name, 'func')
    events_file = sorted([i for i in os.listdir(sub_func_path) if 'events' in i and 'rest' not in i and \
                        int(i.split('-')[-1].split('_')[0])<=12 and 'discard' not in i])

    for model_idx in range(4):
        trial_type_sess = []
        # intial containers
        dtseries_sess = np.zeros((3*n_tr, 91282))
        onset_sess = np.zeros((3*n_event))
        duration_sess = np.zeros((3*n_event))
        for run_idx in range(3):
            run_file_idx = run_idx + model_idx * 3 + 1
            events_file_name = '%s_ses-action01_task-action_run-%02d_events.tsv'%(sub_name, run_file_idx)
            run_name = 'ses-action01_task-action_run-%d'% run_file_idx
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
            # concantenate all info into sess params
            dtseries_sess[n_tr*run_idx:n_tr*(run_idx+1)] = dtseries
            onset_sess[n_event*run_idx:n_event*(run_idx+1)] = onset + run_idx * n_tr * tr
            duration_sess[n_event*run_idx:n_event*(run_idx+1)] = duration
            trial_type_sess.extend(trial_type)
        # prepare design matrix
        events = pd.DataFrame({'trial_type':trial_type_sess, 'onset':onset_sess, 'duration':duration_sess})
        design_matrix = make_first_level_design_matrix(frame_times, events, drift_model=None, hrf_model='spm')
        design_matrix.drop(design_matrix.columns[-1], axis=1, inplace=True)
        # add drift columns
        drift_order = 2
        frame_times_single = np.arange(n_tr) * tr 
        drift_effect = np.zeros((n_tr * 3, 3*(drift_order+1)))
        tmax = float(frame_times_single.max())
        for run_idx in range(3):
            for k in range(drift_order+1):
                drift_effect[n_tr*run_idx:n_tr*(run_idx+1), (drift_order+1)*run_idx+k] = (frame_times_single / tmax) ** k
        drift_effect = pd.DataFrame(drift_effect)
        # concantenate 
        design_matrix = pd.concat([design_matrix.reset_index(drop=True), drift_effect], ignore_index=True, axis=1)  
        # perform GLM
        reg = Ridge(alpha=alpha, fit_intercept=False).fit(design_matrix.values, dtseries_sess[:, :59412])
        beta_clean[model_idx] = reg.coef_[:, :n_class].transpose(1,0).astype(np.float32)
        print('Finish performing GLM in %s %s model %02d'%(sub_name, sess_name, model_idx+1))
    # save data
    if not os.path.exists(pjoin(beta_path, sub_name)):
        os.makedirs(pjoin(beta_path, sub_name))
    np.save(beta_clean_path, beta_clean.astype(np.float32))
