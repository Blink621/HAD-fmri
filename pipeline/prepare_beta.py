import os
import numpy as np
import pandas as pd
import nibabel as nib
from os.path import join as pjoin

# define paths and sub names
events_path = '/nfs/z1/zhenlab/BrainImageNet/action/data/bold/nifti'
out_path = '/nfs/z1/zhenlab/BrainImageNet/action/data/bold/derivatives/beta'
ciftify_dir = '/nfs/z1/zhenlab/BrainImageNet/action/data/bold/derivatives/ciftify'
result_dir = 'MNINonLinear/Results/'
sub_names = sorted([i for i in os.listdir(ciftify_dir) if i.startswith('sub-')])
sub_names_exist = os.listdir(out_path)
# sub_names = list(set(sub_names) - set(sub_names_exist) - set(['sub-08', 'sub-16', 'sub-21', 'sub-24']))

sub_names = ['sub-16']

for sub_name in sub_names:
    # prepare label csv
    sub_events_path = pjoin(events_path, sub_name)
    sub_beta_path = pjoin(out_path, sub_name)
    df_img_name = []
    # find imagenet task
    imagenet_sess = [_ for _ in os.listdir(sub_events_path) if ('action01' in _)]
    # Remember to sort list !!!
    imagenet_sess.sort()
    # loop sess and run
    for sess in imagenet_sess:
        for run in np.linspace(1,12,12, dtype=int):
            # open ev file
            events_file = pjoin(sub_events_path, sess, 'func',
                                '{:s}_{:s}_task-action_run-{:02d}_events.tsv'.format(sub_name, sess, run))
            tmp_df = pd.read_csv(events_file, sep="\t")
            df_img_name.append(tmp_df.loc[:, ['trial_type', 'stim_file']])
    df_img_name = pd.concat(df_img_name)
    df_img_name.columns = ['class_id', 'image_name']
    # make path
    if not os.path.exists(sub_beta_path):
        os.makedirs(sub_beta_path)
    df_img_name.to_csv(pjoin(sub_beta_path, f'{sub_name}_action-label.csv'), index=False)
    print(f'Finish preparing labels for {sub_name}')
            
    # prepare brain voxel data
    # MNINolinear/Results disposit all the runs data
    _result_path = pjoin(ciftify_dir, sub_name, result_dir)
    # extract the ImageNet runs
    imagenet_runs = [_ for _ in os.listdir(_result_path) if ('action01' in _) and int(_.split('_')[0][-1]) < 5]
    imagenet_runs = ['ses-action%02d_task-action_run-%02d' \
                     %(int(x[x.index('action')+6:x.index('_task')]), int(x.split('-')[-1]) ) \
                     for x in imagenet_runs]
    imagenet_runs.sort() # sort() to be [01-10] now
    # imagenet_runs.pop(4)
    # initialize the data array
    imagenet_data = np.zeros((1, 91282))
    # loop run
    for item in imagenet_runs:
        single_run = item.rsplit('-',1)[0] + '-' + str(int(item.rsplit('-',1)[1])) # make 01,02,03 ... to be 1 2 3
        print(single_run)
        # collect session number & run number
        ses = int((single_run.split('_')[0]).replace('ses-action', ''))
        runidx = int(single_run.split('_')[2].replace('run-', ''))

        # prepare .feat/GrayordinatesStats dir
        cope_dir = '{0}/{0}_hp100_s4_level1.feat/GrayordinatesStats'.format(single_run)
        cope_path = pjoin(ciftify_dir, sub_name, result_dir, cope_dir)
        # loop trial 
        for num in list(range(60)):
            cope_file = pjoin(cope_path, 'cope{}.dtseries.nii'.format(num+1))
            dt_data = nib.load(cope_file).get_fdata()
            imagenet_data = np.concatenate((imagenet_data, dt_data), axis=0)
            # stim_resp_map[ev_df.loc[num, 'trial_type']] = dt_data[:,:] #This can replace with roi_mat
            print(f'Finish packing image{num+1} in {sub_name} {single_run}')
    imagenet_data = np.delete(imagenet_data, 0, axis=0)
    # save data      
    np.save(pjoin(sub_beta_path, f'{sub_name}_action-beta.npy'), imagenet_data)
    
    

