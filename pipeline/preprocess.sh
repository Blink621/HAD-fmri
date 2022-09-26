# BIDS Data structure transformation
python data2bids.py /nfs/z1/zhenlab/BrainImageNet/action scaninfo_second.xlsx -q ok --overwrite --skip-feature-validation  

# Copy T1w structure data if it has been done before
python cp_T1w.py

# fMRIprep preprocessing 
export prjdir=/nfs/z1/zhenlab/BrainImageNet/action/data/bold
export bids_fold=$prjdir/nifti
export out_dir=$prjdir/derivatives
export work_dir=$prjdir/workdir
export license_file=/usr/local/neurosoft/freesurfer/license.txt

fmriprep-docker $bids_fold $out_dir participant --skip-bids-validation --participant-label 01 02 03 04 05 06 07 08 09 10 12 13 15 19 20 28 29 30 --fs-license-file $license_file --output-spaces anat fsnative -w $work_dir

# Ciftify: from the volume space to surface space
# ciftify_recon_all
# Remember to change ciftify_recon_all code before running!!!!
# Line 495: if 'v6.' in fs_version:
# Line 594: add '-nc' between T1w_nii and freesurfer_mgz
export SUBJECTS_DIR=/nfs/z1/zhenlab/BrainImageNet/action/data/bold/derivatives/freesurfer 
export CIFTIFY_WORKDIR=/nfs/z1/zhenlab/BrainImageNet/action/data/bold/derivatives/ciftify/
python run_cmd.py /nfs/z1/zhenlab/BrainImageNet/action/ -c "ciftify_recon_all --surf-reg MSMSulc --resample-to-T1w32k sub-<subject>" -s 08 04 05 06 09 01

# ciftify subject fmri: process the volume data in surface space. This will make output in MNINonLinear Results folder
# if ciftify_recon_all doesn't change. This will make error output
python ciftify_subject_fmri.py

# GLM part
# fill nifti info
python prepare_nifti_events.py
# prepare glm files
python prepare_glm_file.py -s 12 13
# run glm
python glm_action.py -s 20 28 29 30 



