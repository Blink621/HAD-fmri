
import os, subprocess, argparse, traceback
from os.path import join as pjoin
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from email.mime.text import MIMEText
from email.header import Header
from smtplib import SMTP_SSL


def send_mail(receiver, mail_title, mail_content):
    host_server = 'smtp.qq.com'
    sender_qq = '1045418215@qq.com'
    pwd = 'dflxxfxcrwkybfeg'
    # ssl
    smtp = SMTP_SSL(host_server)
    # set_debuglevel()
    smtp.set_debuglevel(1)
    smtp.ehlo(host_server)
    smtp.login(sender_qq, pwd)
    msg = MIMEText(mail_content, "plain", 'utf-8')
    msg["Subject"] = Header(mail_title, 'utf-8')
    msg["From"] = sender_qq
    msg["To"] = receiver
    smtp.sendmail(sender_qq, receiver, msg.as_string())
    smtp.quit()


def run_ica(args):
    # collect waiting runs
    fmriprep_dir = pjoin(args.projectdir,'data/bold/derivatives/fmriprep')
    melodic_dir = pjoin(args.projectdir,'data/bold/derivatives/melodic')

    sub = [ "sub-%02d"% int(_) for _ in args.subject]
    sub_dirs = [ _  for _ in os.listdir(fmriprep_dir) if _ in sub]
    ses_dirs = [pjoin(fmriprep_dir, _, __) for _ in sub_dirs for __ in os.listdir(pjoin(fmriprep_dir, _)) if 'ses-action' in __ ]
    func_files = [ __ for _ in ses_dirs for __ in os.listdir(pjoin(_, 'func')) if 'space-T1w_desc-preproc_bold_hp128.nii.gz' in __ ]
    func_files.sort()
    try:
        # run melodic
        for nii in func_files[args.range[0]:args.range[1]]:
            sub_dir = nii[nii.index('sub'):nii.index('_ses')]
            ses_dir = nii[nii.index('ses'):nii.index('_task')]
            ica_output = os.path.join(melodic_dir, sub_dir, ses_dir,nii[nii.index('sub'):nii.index('_space')]+'.ica')
        
            # melodic_command = ' '.join(['melodic', '-i', pjoin(fmriprep_dir, sub_dir, ses_dir,'func', nii), '-o', ica_output,
            #                         '-v --nobet --bgthreshold=1 --tr=2 -d 0 --mmthresh=0.5 --report'])
            ica_output = ica_output+'/filtered_func_data.ica'
            melodic_command = ' '.join(['melodic', '-i', pjoin(fmriprep_dir, sub_dir, ses_dir,'func', nii), '-o', ica_output,
            '-v --debug --nobet --tr=2 --report --Oall'])
            del_timeseries_1 = ' '.join(['imrm', ica_output+'/alldat'])
            del_timeseries_2 = ' '.join(['imrm', ica_output+'/concat_dat'])
            del_command = ';'.join([del_timeseries_1, del_timeseries_2])

            try:
                if not os.path.exists(ica_output):
                    os.makedirs(ica_output)
                    print('os.makedirs({})'.format(ica_output))
                    # subprocess.check_call(melodic_command, shell=True)
                    subprocess.check_call(';'.join([melodic_command, del_command]), shell=True)
                else:
                    print('{} exists!'.format(ica_output))
            except subprocess.CalledProcessError:
                raise Exception('MELODIC: Error happened in file {}'.format(nii))
            
            
            # confound .csv for motion regressor
            if args.motion:
                confoundcsv = pjoin(fmriprep_dir, sub_dir, ses_dir, 'func', \
                    '{}_desc-confounds_timeseries.tsv'.format(nii[nii.index('sub'):nii.index('_space')]))
            # deniose
            if args.denoise:
                # make dtrends & spatial-filter output dir
                mix_orignal_dir = pjoin(ica_output, 'series_original')
                nii_orignal_dir = pjoin(ica_output, 'spatial_original')
                if not os.path.exists(mix_orignal_dir):
                    os.makedirs(mix_orignal_dir)
                    print('os.makedirs({})'.format(mix_orignal_dir))
                if not os.path.exists(nii_orignal_dir):
                    os.makedirs(nii_orignal_dir)
                    print('os.makedirs({})'.format(nii_orignal_dir))
                complete_dirs = [pjoin(ica_output, 'melodic_mix'),\
                                 pjoin(ica_output, 'melodic_FTmix'),\
                                 pjoin(ica_output, 'melodic_IC.nii.gz'),\
                                 pjoin(mix_orignal_dir, 'melodic_FTmix'),\
                                 pjoin(mix_orignal_dir, 'melodic_FTmix'),\
                                 pjoin(nii_orignal_dir, 'melodic_IC.nii.gz')]
                if all([os.path.exists(_) for _ in complete_dirs]):
                    continue
                else:
                    # move orignal files
                    mv_command = [' '.join(['mv', pjoin(ica_output, 'melodic_mix'), mix_orignal_dir]),\
                        ' '.join(['mv', pjoin(ica_output, 'melodic_FTmix'), mix_orignal_dir]), \
                            ' '.join(['mv', pjoin(ica_output, 'melodic_IC.nii.gz'), nii_orignal_dir])]
                    for _ in mv_command:
                        print(_)
                        subprocess.check_call(_, shell=True)
                    # load & create 
                    df_mix = pd.read_csv(pjoin(mix_orignal_dir,'melodic_mix'), sep='  ', header=None)
                    df_ft = pd.read_csv(pjoin(mix_orignal_dir,'melodic_FTmix'), sep='  ', header=None)
                    if df_mix.shape[1]==1:
                        df_mix = pd.read_csv(pjoin(mix_orignal_dir,'melodic_mix'), sep=' ', header=None)
                        df_ft = pd.read_csv(pjoin(mix_orignal_dir,'melodic_FTmix'), sep=' ', header=None)
                    # name the columns
                    df_mix.columns = [ _ for _ in range(df_mix.shape[-1])]
                    df_ft.columns = [ _ for _ in range(df_ft.shape[-1])]
                    # motion
                    df_conf = pd.read_csv(confoundcsv, sep='\t')
                    
                    # output ftmix, rows will change for frequency power
                    df_FT = pd.DataFrame(data=np.zeros((int(np.ceil(df_mix.shape[0]/2))-1,\
                        df_mix.shape[-1])), columns=[_ for _ in range(df_mix.shape[-1])])
                    
                    # prepare regressors matrix
                    # polyfit
                    x = np.linspace(1, df_mix.shape[0], df_mix.shape[0])
                    X = np.vstack(tuple([x**_ for _ in range(args.order+1)]))
                    # motion
                    if args.motion:
                        motion = np.vstack(tuple([np.array(df_conf[_]).astype(np.float64) for _ in ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']]))
                        X = np.vstack((X,motion))
                        X = X.transpose()
                    
                    for __ in range(df_mix.shape[-1]):
                        
                        # dependent variable
                        y = np.array(df_mix[__])
                        
                        # polyfit with motion
                        reg = LinearRegression().fit(X, y)
                        res = y - reg.predict(X)
                
                        # frequency spetrum 
                        signal = np.array(res, dtype=float)
                        power = np.abs(np.fft.fft(signal))**2
                        n = int(signal.size)
                        timestep = 2
                        freqs = np.fft.fftfreq(n, timestep)
                        power = power[freqs>0]
                        freqs = freqs[freqs>0]
                        idx = np.argsort(freqs)
                
                        # write into DataFrame
                        df_mix[__] = np.float32(res)
                        df_FT[__] = np.float32(power[idx])
                        
                        # print('IC {} done'.format(__))
                
                    # generate
                    if not os.path.exists(pjoin(ica_output,'melodic_mix')):
                        df_mix.to_csv(pjoin(ica_output,'melodic_mix'), sep=' ', columns=None, \
                            header=None, index=False)
                        df_FT.to_csv(pjoin(ica_output,'melodic_FTmix'), sep=' ', columns=None, \
                            header=None, index=False)
                    
                    melodic_IC = pjoin(nii_orignal_dir, 'melodic_IC.nii.gz')
                    fslmaths_command = ' '.join(['fslmaths',melodic_IC,'-s', str(args.fwhm/2.355), pjoin(ica_output, 'melodic_IC.nii.gz')])
                    subprocess.check_call(fslmaths_command, shell=True)
                    print(fslmaths_command)
        if args.email_address:
            send_mail(args.email_address, 'congratulations', 'ICA {}-{}sucessfully done'.format(args.range[0],args.range[1]))
    except Exception:
        print(Exception)
        if args.email_address:
            send_mail(args.email_address, 'sorry', 'ICA {}-{} has some problems as follows:\n {}\n{}'\
                      .format(args.range[0],args.range[1],traceback.print_exc(),traceback.format_exc()))

# %%
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("projectdir", help="path of project dir")
    parser.add_argument("range", type=int, nargs='+')
    parser.add_argument("--subject", nargs="+", help="subject filter")
    parser.add_argument("--denoise", action="store_true", help="if chosen, spatial & temporal denoise will be put on ICs")
    parser.add_argument("--order", type=int, help="the order of polyfit, practically it should be less than half of the run duration.",default=4)
    parser.add_argument("--motion", action="store_true", help="if chosen, motion regressor will be added in to denoise")
    parser.add_argument("--fwhm", type=float, help="the full width at the half maximum",default=4)
    parser.add_argument("--email-address", help="if given, a message will be sent to the address once the procedure succeeds or goes wrong")
    args = parser.parse_args()

    run_ica(args)

    # class args:
    #     range=[2,150]
    #     order=4
    #     fwhm=4
    #     projectdir='/nfs/m1/BrainImageNet/NaturalObject'
    #     subject=['core02','core03']
    #     denoise=True
    #     motion=True
    #     email_address='1045418215@qq.com'
    # python melodic_denoise.py /nfs/z1/zhenlab/BrainImageNet/action 0 500 --subject 1 2 3 4 5 --email-address 1458940886@qq.com
