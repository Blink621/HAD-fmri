import os
import subprocess
import numpy as np
from os.path import join as pjoin

melodic_path = '/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/derivatives/melodic'
os.chdir(melodic_path)
subs = [_ for _ in os.listdir('./') if 'sub-01' in _]
ses_flag = 'ImageNet01' # if select all sessions, choose 'ses' 
sub_ses = [pjoin(sub, ses) for sub in subs for ses in os.listdir(sub) \
     if ses_flag in ses]
sub_ses.sort()

for folder in sub_ses[0:1]:
    ica_folders = [_ for _ in os.listdir(folder) if '.ica' in _ and 'natural' in _]
    ica_folders.sort()
    sub = folder.split('/')[0]
    ciftify_root  = '/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/derivatives/ciftify/' + sub + '/T1w'
    fmriprep_root = pjoin('/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/derivatives/fmriprep',folder,'func') 
    for ica_folder in ica_folders[0:10]:
        print(ica_folder)
        # ===============================
        # prep the Movement_Regressor.txt 
        # ===============================
        source_file   = ica_folder.replace('.ica', '_confounds.txt').replace(sub+'_', '')
        source_MotReg = pjoin(melodic_path, 'MotionRegressor', source_file)
        dest_MogReg   = pjoin(folder, ica_folder, 'Movement_Regressors.txt')
        if not os.path.exists(dest_MogReg):
            MotReg_cmd = 'ln -s %s %s' % (source_MotReg, dest_MogReg)
            print("\033[0;31;40m Creating Movement_Regressor.txt\033[0m")
            subprocess.check_call(MotReg_cmd, shell=True)
        else: print("\033[0;34;40m Movement_Regressor.txt exists! \033[0m")
        # IN sub/ses/.ica
        os.chdir(pjoin(folder, ica_folder))
        # ===============================
        # check if exists the naccessary files in xxx.ica
        # if melodic_IC.nii.gz exists files we need will exists
        # ===============================
        melodic_flag = 'filtered_func_data.ica/melodic_IC.nii.gz'
        if os.path.exists(melodic_flag):
            # prepare mask & mean_func
            softlink_cmds = '/usr/local/neurosoft/fsl/bin/imln filtered_func_data.ica/mask mask; /usr/local/neurosoft/fsl/bin/imln filtered_func_data.ica/mean mean_func'
            print("\033[0;31;40m Creating soft links to mean & mask\033[0m")
            subprocess.check_call(softlink_cmds, shell=True)
            # make other folders and files
            mkdir_cmds = 'mkdir -p mc; mkdir -p reg'
            crtpar_cmd = """ cat Movement_Regressors.txt | awk '{ print $4 " " $5 " " $6 " " $1 " " $2 " " $3}' > mc/prefiltered_func_data_mcf.par """
            print("\033[0;31;40m Creating mc reg mcf.par\033[0m")
            subprocess.check_call(';'.join([mkdir_cmds, crtpar_cmd]), shell=True)
        # ===============================
        # prepare functional file
        # ===============================
        which_imln  = '/usr/local/neurosoft/fsl/bin/imln' 
        func_file   = pjoin(fmriprep_root, ica_folder.replace('.ica', '_space-T1w_desc-preproc_bold_hp128'))
        target_file = pjoin(melodic_path, folder, ica_folder, 'filtered_func_data')
        lnfunc_cmd  = ' '.join([which_imln, func_file, target_file])
        print("\033[0;31;40m Soft link to func file\033[0m")
        subprocess.check_call(lnfunc_cmd, shell=True)
        # ===============================
        # prepare structural files
        # ===============================
        T1w_cmd = '/usr/local/neurosoft/fsl/bin/imln %s reg/highres' % (pjoin(ciftify_root, 'T1w'))
        wm_cmd  = '/usr/local/neurosoft/fsl/bin/imln %s reg/wmparc' % (pjoin(ciftify_root, 'wmparc'))
        ex_cmd  = '/usr/local/neurosoft/fsl/bin/imln %s reg/example_func' % pjoin(os.getcwd(), 'mean_func')
        mat_cmd = '/usr/local/neurosoft/fsl/bin/makerot --theta=0 > reg/highres2example_func.mat'
        print("\033[0;31;40m Creating reg files\033[0m")
        subprocess.check_call(';'.join([T1w_cmd, wm_cmd, ex_cmd, mat_cmd]), shell=True)
        # ===============================
        # then make a softlink at the ciftify/ folder
        # ===============================
        cd_cmd         = 'cd %s' % pjoin(melodic_path, folder)
        t1w_result_ses = pjoin(ciftify_root, 'Results', ica_folder.replace('.ica','').replace(sub+'_',''))
        mk_cmd         = 'mkdir -p %s' % (t1w_result_ses)
        ln_cmd         = 'ln -s %s %s' % (pjoin(melodic_path, folder, ica_folder), pjoin(t1w_result_ses, ica_folder.replace(sub+'_','')))
        if not os.path.exists(pjoin(t1w_result_ses, ica_folder.replace(sub+'_',''))):
            print("\033[0;31;40m Link .ica folder\033[0m")
            subprocess.check_call(';'.join([cd_cmd, mk_cmd, ln_cmd]), shell=True)
        else: print("\033[0;34;40m .ica folder exists! \033[0m")
        # back to melodic path
        os.chdir(melodic_path)
        # ===============================
        # run fix
        # ===============================
        which_fix = '/usr/local/neurosoft/fix/fix'
        which_ica = pjoin(melodic_path, folder, ica_folder)
        training  = pjoin('/usr/local/neurosoft/fix', 'training_files/HCP_hp2000.RData')
        fix_thres = '30'
        high_pass = '-h 128'
        fix_cmd   = ' '.join([which_fix, which_ica, training, fix_thres, high_pass])
        print("\033[0;31;40m Run fix...\033[0m")
        subprocess.check_call(fix_cmd, shell=True)
        # # ===============================
        # # move and rename files at rhe cifity file
        # # ===============================
        # which_immv   = '/usr/local/neurosoft/fsl/bin/immv'
        # cleaned_file = pjoin(t1w_result_ses, ica_folder.replace(sub+'_',''), 'filtered_func_data_clean')
        # target_file  = pjoin(t1w_result_ses, ica_folder.replace(sub+'_','').replace('.ica', '_clean'))
        # cleanmv_cmd  = ' '.join([which_immv, cleaned_file, target_file])
        # print("\033[0;31;40m Move cleaned data \033[0m")
        # subprocess.check_call(cleanmv_cmd, shell=True)
        # # ===============================
        # # remove the unrecognizible file
        # # ===============================
        # which_imrm   = '/usr/local/neurosoft/fsl/bin/imrm'
        # dorm_cmd     = ' '.join([which_imrm, cleaned_file])
        # print("\033[0;31;40m Remove unrecognizible file \033[0m")
        # subprocess.check_call(dorm_cmd, shell=True)



