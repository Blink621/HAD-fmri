import os
import nibabel as nib
import numpy as np
import pandas as pd
from os.path import join as pjoin
import scipy.io as sio

events_path = '/nfs/z1/userhome/ZhouMing/workingdir/BIN/action/HAFD'
beh_path = '/nfs/z1/zhenlab/BrainImageNet/action/data/behavior/'
fmriprep_dir = '/nfs/z1/zhenlab/BrainImageNet/action/data/bold/derivatives/fmriprep'

# sub_special_order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 5] 
# sub_names = sorted([i for i in os.listdir(fmriprep_dir) if i.startswith('sub-') and not i.endswith('.html')])
sub_names = ['sub-01']

for sub_name in sub_names:
    # 
    sub_beh_name = 'sub{:s}'.format(sub_name[-2:])
    sub_events_path = pjoin(events_path, sub_name)
    # find imagenet task
    sess_names = [_ for _ in os.listdir(sub_events_path) if ('action' in _) and ('5' not in _) and ('6' not in _)]
    # loop sess and run
    for sess in sess_names:
        for run in np.linspace(1,12,12, dtype=int):
            sess_beh = f'sess{sess[-2:]}'
            # if run == 11:
            #     run_beh = 1  
            # else:
            #     run_beh = run
            # Note
            # if there is no problem in during exp, run_exp = run
            # if the run number adjusted, you can change run_beh like sub_special_order
            run_beh = run
            # open ev file
            events_file = pjoin(sub_events_path, sess, 'func',
                                '{:s}_{:s}_task-action_run-{:02d}_events.tsv'.format(sub_name, sess, run))
            ev_df =  pd.read_csv(events_file, sep='\t')
            # delete to-do columns
            delete_name = 'TODO -- fill in rows and add more tab-separated columns if desired'
            if delete_name in ev_df.columns:
                ev_df.drop([delete_name], axis=1, inplace=True)
            # # open beh data
            # beh_file = pjoin(beh_path, sub_beh_name, sess_beh,
            #                  '{:s}_{:s}_run{:02d}.mat'.format(sub_beh_name, sess_beh, run_beh))
            # design_file = pjoin(beh_path, sub_beh_name, sess_beh,
            #                  '{:s}_{:s}_design.mat'.format(sub_beh_name, sess_beh))
            # # handle the situation that beh_file not existed
            # if not os.path.exists(beh_file):
            #     print(f'Can not find {beh_file}. Filling nifti events using design file')
            #     design_mat = sio.loadmat(design_file)
            #     sess_par = design_mat['sessPar'][:, run-1, :]
            #     # info: column0: onset; column1: duration; column2: trial type; column4: rt
            #     info = np.zeros((sess_par.shape[0], 4))
            #     info[:, [0, 1, 2]] = sess_par[:, [0, 2, 1]]
            #     # change the duration to be 2s
            #     info[:, 1] = 2
            #     # prepare stim_file
            #     stim_tmp = design_mat['sessStim'][:, run_beh-1]
            # else:
            #     beh_mat = sio.loadmat(beh_file)
            #     # save beh onset, duration, trial_type, response_time and stim_file into ev
            #     trial = beh_mat['trial']
            #     # info: column0: onset; column1: duration; column2: trial type; column4: rt
            #     info = np.zeros((trial.shape[0], 4))
            #     info = trial[:, [5, 2, 1, 4]] 
            #     # change the duration to be the real finish time substract real present time
            #     info[:, 1] = trial[:, 6] - trial[:, 5]
            #     # prepare stim_file
            #     stim_tmp = beh_mat['sessStim'][:, run_beh-1]
            # if ev_df.empty:
            #     tmp_df = pd.DataFrame(info, columns=['onset', 'duration', 'trial_type', 'response_time'])
            #     ev_df = pd.concat([ev_df, tmp_df])
            # else:
            #     ev_df.loc[:, ['onset', 'duration', 'trial_type', 'response_time']] = info
            # # merge stim_file into nifti events
            # stim_file = [x[0] for x in stim_tmp]
            # ev_df['stim_file'] = stim_file
            ev_df.to_csv(events_file, index=False, sep='\t')
            print('Finish preparing events in {:s}_{:s}_run-{:02d}'.format(sub_name, sess, run))
                
