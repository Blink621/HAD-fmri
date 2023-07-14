# A large-scale fMRI dataset for human action recognition
Human Action Dataset (HAD), a large-scale functional magnetic resonance imaging (fMRI) dataset for human action recognition. HAD contains fMRI responses to 21,600 video clips from 30 participants. The video clips encompass 180 human action categories and offer a comprehensive coverage of complex activities in daily life. The dataset contains raw data, derived data from fMRIPrep and Ciftify, and suface-based analyzed data.   

To get more details, please refer to the paper at https://www.nature.com/articles/s41597-023-02325-6 and the dataset at https://openneuro.org/datasets/ds004488

## Preprocess procedure
The MRI data were first converted into the Neuroimaging Informatics Technology Initiative (NIfTI) format and then organized into the Brain Imaging Data Structure (BIDS)using HeuDiConv (https://github.com/nipy/heudiconv). Then fMRIprep 20.2.1 were used to perform volume prepocess. Detailed information on fMRIprep pipelines can be found in the online documentation of the fMRIPrep(https://fmriprep.org). Then, all the preprocessed individual fMRI data were registered onto the 32k fsLR space using the Ciftify toolbox.

## Volume-based process
**code: ./volume_preprocess/volume_preprocess.sh**

The data2bids.py helps you to reorganize the data structure to BIDS. You have to first prepare the scan_info.xlsx to fit for your experiment protocol and design. 
fMRIprep were performed using docker and detailed usage notes are available in codes, please read carefully and modify variables to satisfy your customed environment.

## Surface-based process
**code: ./surface_preprocess/surface_preprocess.sh**

The cifify_recon_all function was used to register and resample individual surfaces to 32k standard fsLR surfaces via surface-based alignment. The ciftfy_subject_fmri function was then used to project functional MRI data onto the fsLR surface. 
In most circumstances the only necessary operation it to change the path to dataset, other optional settings are explained by annotations.

## Validation
Some neccessary auxilary files have been stored in the *supportfiles* folder.
Additional results files have been stored in the *results* folder.
For computational convenience, some intermediate files will be generated by codes.
### Behavior information

**code: ./validation/behavior.ipynb**

During the scanning, participants were asked to press one of two response buttons as quickly as possible after a clip disappeared to indicate that the human action presented in the clip was a sport or a non-sport action. The corresponding response information (i.e., accuracy and response rate) were analyzed for each participant. 

### Framewise displacement

**code: ./validation/FD.ipynb**

The head motion of the participants was quantified with the framewise displacement (FD) metric, which measures instantaneous head motion by comparing the motion between the current and the previous volume. The FD were computed and visualized across all scanning run for each participant.  

### General linear model

**code: ./validation/GLM.py**

Customed GLM analyses were conducted on the surface data to deconvovle the hemodynamic effects of BOLD signal. As the 180 action categories were cycled once every three runs, we modeled the data from each cycle to estimate the BOLD responses to each category. The vertex-specific responses (i.e., beta values) estimated for each clip were used for further analyses.

### Contrast-to-noise ratio(CNR)

**code: ./validation/CNR.py**

A contrast-to-noise ratio (CNR) analysis was performed to check if the HACS clips can induce desired signal changes in each vertex across the cortical surface. The CNR was calculated as the averaged beta values across all stimuli divided by the temporal standard deviation of the residual time series from GLM models. For individual maps of CNR, please see results under the *results/brain_map_individual/cnr* folder.

### Test-retest reliability

**code: ./validation/reliability.py**

Next, we assessed the test-retest reliability of BOLD responses for the 180 action categories. As the 180 action categories were repeated four times by cycling every three runs in each session, we computed the Pearson correlation between the brain responses of the 180 categories from the odd and even cycles within each participant to measure the test-retest reliability. For individual maps of reliability, please see results under the *results/brain_map_individual/reliability* folder.

### Inter-subject correlation(ISC)

**code: ./validation/ISC.py**

An inter-subject correlation (ISC) analysis was performed to validate that our dataset can reveal consistent action category-selective response profiles across participants. Here, the ISC is measured for each participant by calculating the Pearson correlation between her/his category-specific response profiles (i.e., beta series) with the averaged category-specific response profiles from the remaining 29 participants.

### Representation similarity analysis(RSA)

**code: ./validation/RSA.ipynb**

A representational similarity analysis (RSA) was conducted to validate that multi-voxel activity patterns from the data represent a rich semantic structure of action categories. Specifically, the representational dissimilarity matrix (RDM) of the 180 categories was constructed by computing the Pearson correlation between the multi-voxel activity patterns from each category in different visual pathways. The RDMs from different visual pathways were quantitatively evaluated by the Spearman correlation among them.