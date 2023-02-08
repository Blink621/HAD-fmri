import os
import subprocess
import numpy as np
import pandas as pd
from os.path import join as pjoin

# define path
store_path = '/nfs/z1/userhome/ZhouMing/workingdir/BIN/action/data/bold/'
dataset_path = '/nfs/z1/userhome/ZhouMing/workingdir/BIN/action/HAD'
org_stim_path = '/nfs/z1/userhome/ZhouMing/workingdir/BIN/action/exp/video'
# sub_names = ['sub-%02d'%(i+1) for i in range(30)]
sub_names = ['sub-01']

for sub_name in sub_names:
    # # copy raw data
    # # implement by superlink
    # raw_data_path = pjoin(store_path, 'nifti', sub_name)
    # link_raw_data_path = pjoin(dataset_path, sub_name)
    # if not os.path.exists(link_raw_data_path):
    #     link_raw_data_cmd = f'cp -r {raw_data_path} {link_raw_data_path}'
    #     subprocess.call(link_raw_data_cmd, shell=True)

    # # copy preprocessed data
    # preprocessed_path = pjoin(store_path, f'derivatives/fmriprep/{sub_name}/ses-action01/func/')
    # link_preprocessed_path = pjoin(dataset_path, 'derivatives/fmriprep', sub_name)
    # if not os.path.exists(link_preprocessed_path):
    #     os.makedirs(link_preprocessed_path)
    #     bold_file = ['%s/%s_ses-action01_task-action_run-%d_space-T1w_desc-preproc_bold.nii.gz'%(preprocessed_path, sub_name, i+1) for i in range(12)]
    #     bold_file = ' '.join(bold_file)
    #     confound_file = ['%s/%s_ses-action01_task-action_run-%d_desc-confounds_timeseries.tsv'%(preprocessed_path, sub_name, i+1) for i in range(12)]
    #     confound_file = ' '.join(confound_file)
    #     # copy cmd
    #     cp_preprocessed_cmd = f'cp -l {bold_file} {confound_file} {link_preprocessed_path}'
    #     subprocess.call(cp_preprocessed_cmd, shell=True)

    # copy stimuli and change video path name
    sub_beh_name = 'sub{:s}'.format(sub_name[-2:])
    sub_events_path = pjoin(dataset_path, sub_name)
    # find imagenet task
    sess = 'ses-action01'
    # loop sess and run
    for run in np.linspace(1,12,12, dtype=int):
        sess_beh = f'sess{sess[-2:]}'
        # open ev file
        events_file = pjoin(sub_events_path, sess, 'func',
                            '{:s}_{:s}_task-action_run-{:02d}_events.tsv'.format(sub_name, sess, run))
        ev_df =  pd.read_csv(events_file, sep='\t')
        # delete to-do columns
        delete_name = 'TODO -- fill in rows and add more tab-separated columns if desired'
        if delete_name in ev_df.columns:
            ev_df.drop([delete_name], axis=1, inplace=True)
        # iterate to copy stim into stimuli folder
        stim_files = ev_df['stim_file'].to_list()
        for clip_idx, clip_name in enumerate(stim_files):
            if 'stimuli' not in clip_name:
                stim_file = pjoin(clip_name.split('_')[1], clip_name)
                stim_path = pjoin(org_stim_path, stim_file)
                link_stim_path = pjoin(dataset_path, 'stimuli', stim_file)
                class_path = pjoin(dataset_path, 'stimuli', clip_name.split('_')[1])
                if not os.path.exists(class_path):
                    os.makedirs(class_path)
                # change stim name into csv
                ev_df.at[clip_idx, 'stim_file'] = pjoin(stim_file)
            else:
                stim_path = pjoin(org_stim_path, clip_name[8:])
                link_stim_path = pjoin(dataset_path, clip_name)
                # change stim name into csv
                ev_df.at[clip_idx, 'stim_file'] = clip_name[8:]
            link_stim_cmd = f"cp '{stim_path}' '{link_stim_path}'"
            subprocess.call(link_stim_cmd, shell=True)
        ev_df.to_csv(events_file, index=False, sep='\t')
        print('Finish copying %s %s run%02d'%(sub_name, sess, run))