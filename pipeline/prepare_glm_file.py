import argparse

#%% prepare EVs
def prepare_EVs(args):
    import os
    import glob
    import pandas as pd
    
    for subj in args.subject:
        sub_name = 'sub-{:s}'.format(subj)
        ses = 'ses-action0*' #'ses-*'
        run = 'run-*'
        begin_dur = 12
        event_files = '/nfs/z1/zhenlab/BrainImageNet/action/data/bold/nifti/{0}/{1}/func/{0}_{1}_task-action_{2}_events.tsv'
        event_files = event_files.format(sub_name, ses, run)
        event_files = sorted(glob.glob(event_files))
        trg_dirs = '/nfs/z1/zhenlab/BrainImageNet/action/data/bold/derivatives/ciftify/{0}/MNINonLinear/Results/{1}/EVs'
    
        for ev_file in event_files:
            print(ev_file)
            run_name = os.path.basename(ev_file).split('_')
            run_name[-2] = 'run-' + str(int(run_name[-2].split('-')[-1])) # make 'run-01' become 'run-1'
            run_name = '_'.join([i for i in run_name if i.startswith('ses-') or i.startswith('task-') or i.startswith('run-')])
            trg_dir = trg_dirs.format(sub_name, run_name)
            if not os.path.exists(trg_dir.replace('/EVs', '')):
                continue
            if not os.path.isdir(trg_dir):
                os.mkdir(trg_dir)
            ev_df = pd.read_csv(ev_file, sep='\t')
            for i, onset in enumerate(ev_df['onset']):
                with open(os.path.join(trg_dir, f'trial{i+1}.txt'), 'w') as wf:
                    wf.write('\t'.join([str(onset+begin_dur), str(ev_df['duration'][i]), '1']))
    
#%% 
def check_for_fsf(args):
    import os
    import nibabel as nib
    from os.path import join as pjoin
    
    subj_par = '/nfs/z1/zhenlab/BrainImageNet/action/data/bold/derivatives/ciftify'
    subj_ids = sorted([i for i in os.listdir(subj_par) if i.startswith('sub-') and int(i[-2:])>10 ])
    for subj_id in subj_ids:
        run_par = pjoin(subj_par, subj_id, 'MNINonLinear/Results')
        runs = sorted([i for i in os.listdir(run_par) if 'action' in i])
        for run in runs:
            run_dir = pjoin(run_par, run)
    
            # do some assertions
            nii = nib.load(pjoin(run_dir, run + '.nii.gz'))
            n_vol = nii.header.get_data_shape()[-1]
            tr = nii.header['pixdim'][4]
            n_ev = len(os.listdir(pjoin(run_dir, 'EVs')))
            if n_vol != 156:
                print('n_vol:', n_vol, run_dir)
            if tr != 2:
                print('TR:', tr, run_dir)
            if n_ev != 60:
                print('n_ev:', n_ev, run_dir)
    
# %% 
def prepare_fsf(args):
    import os
    import numpy as np
    from os.path import join as pjoin
    import nibabel as nib
    
    main_par = '/nfs/z1/zhenlab/BrainImageNet/action/data/bold/derivatives/ciftify'
    pipeline_path = '/nfs/z1/zhenlab/BrainImageNet/action/utils/pipeline'
    for subj in args.subject:
        sub_name = 'sub-{:s}'.format(subj)
        run_par = pjoin(main_par, sub_name, 'MNINonLinear/Results')
        runs = sorted([i for i in os.listdir(run_par) if 'action' in i and 'discard' not in i])
        # set params
        high_pass = 100
        n_EVs = 60
        title_EVs = 'trial'
        
        for run in runs:
            run_dir = pjoin(run_par, run)
            # change volume and tr info according to nii.gz file
            nii = nib.load(pjoin(run_dir, run + '.nii.gz'))
            n_vol = nii.header.get_data_shape()[-1]
            tr = int(nii.header['pixdim'][4])
              
            # prepare ev and contrast 
            ev_content = ''
            contrast_content = ''
            # define ev interface info
            # the ev_title and ev_file need to be specific
            for ev in np.linspace(1,n_EVs,n_EVs,dtype=np.int):
                ev_title = f'# EV {ev} title\nset fmri(evtitle{ev}) "{title_EVs}{ev}"\n\n'
                waveform_shape = f'# Basic waveform shape (EV {ev})\n# 0 : Square\n# 1 : Sinusoid\n# 2 : Custom (1 entry per volume)\n' + \
                                  f'# 3 : Custom (3 column format)\n# 4 : Interaction\n# 10 : Empty (all zeros)\nset fmri(shape{ev}) 3\n\n'
                convolution = f'# Convolution (EV {ev})\n# 0 : None\n# 1 : Gaussian\n# 2 : Gamma\n# 3 : Double-Gamma HRF\n' + \
                              f'# 4 : Gamma basis functions\n# 5 : Sine basis functions\n# 6 : FIR basis functions\nset fmri(convolve{ev}) 3\n\n'
                convolve_phase = f'# Convolve phase (EV {ev})\nset fmri(convolve_phase{ev}) 0\n\n'
                temporal_filtering = f'# Apply temporal filtering (EV {ev})\nset fmri(tempfilt_yn{ev}) 1\n\n'
                temporal_derivative = f'# Add temporal derivative (EV {ev})\nset fmri(deriv_yn{ev}) 0\n\n'
                ev_file = f'# Custom EV file (EV {ev})\nset fmri(custom{ev}) "{run_dir}/EVs/{title_EVs}{ev}.txt"\n\n'
                orthogonalise_info = ''
                for ev_ort in range(n_EVs+1):
                    orthogonalise_info += f'# Orthogonalise EV {ev} wrt EV {ev_ort}\nset fmri(ortho{ev}.{ev_ort}) 0\n\n'
                # combine different info
                ev_content += ev_title + waveform_shape + convolution + convolve_phase + temporal_filtering + \
                              temporal_derivative + ev_file + orthogonalise_info
                              
            # define contrast interface info
            # prepare real contrast info
            con_real = ''
            for ev in np.linspace(1,n_EVs,n_EVs,dtype=np.int):
                con_real_pre = f'# Display images for contrast_real {ev}\nset fmri(conpic_real.{ev}) 1\n\n' + \
                                f'# Title for contrast_real {ev}\nset fmri(conname_real.{ev}) "{title_EVs}{ev}"\n\n'
                con_real_vect = ''
                for element in np.linspace(1,n_EVs,n_EVs,dtype=np.int):
                    content = 1 if element == ev else 0 # make diagonal matrix
                    con_real_vect += f'# Real contrast_real vector {ev} element {element}\nset fmri(con_real{ev}.{element}) {content}\n\n' 
                con_real_test = f'# F-test 1 element {ev}\nset fmri(ftest_real1.{ev}) 0\n\n'
                con_real += con_real_pre + con_real_vect + con_real_test
            # prepare original contrast info
            con_orig = ''
            for ev in np.linspace(1,n_EVs,n_EVs,dtype=np.int):
                con_orig_pre = f'# Display images for contrast_orig {ev}\nset fmri(conpic_orig.{ev}) 1\n\n' + \
                                f'# Title for contrast_orig {ev}\nset fmri(conname_orig.{ev}) "{title_EVs}{ev}"\n\n'
                con_orig_vect = ''
                for element in np.linspace(1,n_EVs,n_EVs,dtype=np.int):
                    content = 1 if element == ev else 0 # make diagonal matrix
                    con_orig_vect += f'# Real contrast_orig vector {ev} element {element}\nset fmri(con_orig{ev}.{element}) {content}\n\n' 
                con_orig_test = f'# F-test 1 element {ev}\nset fmri(ftest_orig1.{ev}) 0\n\n'
                con_orig += con_orig_pre + con_orig_vect + con_orig_test
            # prepare mask contrast info
            mask_test = ''
            for ev in np.linspace(1,n_EVs,n_EVs,dtype=np.int):
                for idx in np.linspace(1,n_EVs,n_EVs,dtype=np.int):
                    if idx != ev:
                        mask_test += f'# Mask real contrast/F-test {ev} with real contrast/F-test {idx}?\nset fmri(conmask{ev}_{idx}) 0\n\n' 
            con_mask = '# Contrast masking - use >0 instead of thresholding?\nset fmri(conmask_zerothresh_yn) 0\n\n'
            # combine different info
            contrast_content += con_real + con_orig + con_mask + mask_test
            
            # prepare confounds info
            # confound_file = sub_id.replace('core', '') + '_' + run + '_confounds.txt'
            # confound_begin = '# Add confound EVs text file\n'
            # confound_content = confound_begin + 'set fmri(confoundevs) 1\n\n# Confound EVs text file for analysis 1\nset confoundev_files(1) "/nfs/z1/zhenlab/BrainImageNet/action/data/bold/Analysis_derivatives/ciftify/confounds/' \
            # + confound_file +'"\n'
    
            # remove line break in the end of file
            ev_content = ev_content.strip('\n')
            contrast_content = contrast_content.strip('\n')
            
            # change input data and output dir
            input_data = f'../{run}.nii.gz'
            output_dir = f'{run}_hp{high_pass}_s4_level1'
            fsf_template = open(pjoin(pipeline_path, 'level1.fsf')).read()
            fsf_text = fsf_template.replace('input_data_mark', input_data).replace('output_dir_mark', output_dir) \
                                    .replace('# ev_content', ev_content).replace('# contrast_content', contrast_content) \
                                    .replace('set fmri(confoundevs) 0\n', '')\
                                    .replace('n_evs_orig', str(n_EVs)).replace('n_evs_real', str(n_EVs)) \
                                    .replace('n_con_orig', str(n_EVs)).replace('n_con_real', str(n_EVs)) \
                                    .replace('tr_mark', str(tr)).replace('volumes_mark', str(n_vol)) \
                                    .replace('high_pass_mark', str(high_pass))   
                                    # .replace('# Add confound EVs text file', confound_content)\
                                        
            open(pjoin(run_dir, f'{run}_hp200_s4_level1.fsf'), 'w').write(fsf_text) 
            print('Finished:', run_dir)

#%%
def rename_file(args):
    import glob
    import subprocess
    
    for subj in args.subject:
        sub_name = 'sub-{:s}'.format(subj)
        ses = 'ses-action*'
        run = 'run-*'
        old_files = '/nfs/z1/zhenlab/BrainImageNet/action/data/bold/derivatives/ciftify/{0}' \
                    '/MNINonLinear/Results/{1}_task-action_{2}/' \
                    '{1}_task-action_{2}_Atlas_s0.dtseries.nii'.format(sub_name, ses, run)
        # old_files = '/nfs/z1/zhenlab/BrainImageNet/action/data/bold/derivatives/ciftify/{0}' \
        #             '/MNINonLinear/Results/{1}_task-action_{2}/' \
        #             '{1}_task-action_{2}_hp100_s4_level1.fsf'.format(ses, run)
        old_files = sorted(glob.glob(old_files))
        for old_file in old_files:
            new_file = old_file.replace('_s0', '')
            cmd = ' '.join(['mv -v', old_file, new_file])
            try:
                subprocess.check_call(cmd, shell=True)
            except subprocess.CalledProcessError:
                raise Exception(f'mv: Error happened at {old_file}')
                
                
if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', "--subject", type=str, nargs="+", help="subject id")    
    # parser.add_argument("r",nargs='+',type=int,help='range of')
    args = parser.parse_args()
    prepare_EVs(args)
    # check_for_fsf(args)
    prepare_fsf(args)
    rename_file(args)
