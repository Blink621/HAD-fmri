import argparse

#%%
def run_glm(args):
    import os
    import subprocess
    from os.path import join as pjoin

    # Location of Subject folders (named by subjectID)
    StudyFolder = '/nfs/z1/zhenlab/BrainImageNet/action/data/bold/derivatives/ciftify'
    
    for sub_id in args.subject:
        # subject ID
        Subject = 'sub-{:02d}'.format(int(sub_id))
        ses = 'ses-action'
        ResultsFolder = pjoin(StudyFolder, Subject, 'MNINonLinear/Results')
    
        # If LevelOneTasks is not None, 'ses' will be ignored!
        LevelOneTasks = sorted([i for i in os.listdir(ResultsFolder) \
            if ('discard' not in i) and ('rest' not in i) and (ses in i)])#[-3:]
        LevelOneTasks = '@'.join(LevelOneTasks)
        LevelOneFSFs = LevelOneTasks
        LevelTwoTask = "NONE"
        LevelTwoFSF = "NONE"
    
        # 32 if using HCP minimal preprocessing pipeline outputs
        LowResMesh = '32'
    
        # 2mm if using HCP minimal preprocessing pipeline outputs
        GrayOrdinatesResolution = '2'
    
        # 2mm if using HCP minimal preprocessing pipeline outputes
        OriginalSmoothingFWHM = '2'
    
        # 2mm is no more smoothing (above minimal preprocessing pipelines grayordinates smoothing).
        # Smoothing is added onto minimal preprocessing smoothing to reach desired amount
        FinalSmoothingFWHM = '4'
    
        # File located in ${SubjectID}/MNINonLinear/Results/${fMRIName} or NONE
        # Confound = [i+'_confounds.txt' for i in LevelOneTasks.split('@')]
        # Confound = '@'.join(Confound)
        Confound = "NONE"
        # Use 2000 for linear detrend, 200 is default for HCP task fMRI
        TemporalFilter = "100"
    
        # YES or NO. CAUTION: Only use YES if you want unconstrained volumetric blurring of your data,
        # otherwise set to NO for faster, less biased, and more senstive processing
        # (grayordinates results do not use unconstrained volumetric blurring and are always produced).
        VolumeBasedProcessing = "NO"
    
        # Use NONE to use the default surface registration
        RegName = "NONE"
    
        # Use NONE to perform dense analysis,
        # non-greyordinates parcellations are not supported because they are not valid for cerebral cortex.
        # Parcellation superseeds smoothing (i.e. smoothing is done)
        Parcellation = "NONE"
    
        # Absolute path the parcellation dlabel file
        ParcellationFile = "NONE"
    
        cmd = ' '.join([
            '${HCPPIPEDIR}/TaskfMRIAnalysis/TaskfMRIAnalysis.sh',
            '--path=' + StudyFolder,
            '--subject=' + Subject,
            '--lvl1tasks=' + LevelOneTasks,
            '--lvl1fsfs=' + LevelOneFSFs,
            '--lvl2task=' + LevelTwoTask,
            '--lvl2fsf=' + LevelTwoFSF,
            '--lowresmesh=' + LowResMesh,
            '--grayordinatesres=' + GrayOrdinatesResolution,
            '--origsmoothingFWHM=' + OriginalSmoothingFWHM,
            '--confound=' + Confound,
            '--finalsmoothingFWHM=' + FinalSmoothingFWHM,
            '--temporalfilter=' + TemporalFilter,
            '--vba=' + VolumeBasedProcessing,
            '--regname=' + RegName,
            '--parcellation=' + Parcellation,
            '--parcellationfile=' + ParcellationFile
        ])
        print(cmd)
        try:
            subprocess.check_call(cmd, shell=True)
        except subprocess.CalledProcessError:
            raise Exception('GLM: Error happened in subject {}'.format(Subject))


if __name__ == '__main__':
    # prepare_EVs()
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', "--subject", type=str, nargs="+", help="subject id")    
    # parser.add_argument("r",nargs='+',type=int,help='range of')
    args = parser.parse_args()
    run_glm(args)

