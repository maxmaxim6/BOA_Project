import os,glob,stat
import pandas as pd
import numpy as np
import re

user_input = "/mnt/d/temp/University/masters/thesis/data/temu2016/"
file_name = "temu2016_ustc2016_mta_starto2017_iscx2016"
files_path = os.path.join('/mnt/d/temp/University/masters/thesis/data/temu2016/',file_name+'.csv')
first=True
os.chdir(user_input)
with open(files_path,'a') as save_file:
    for folder in glob.glob("*/"):
        os.chdir(os.path.join(user_input,folder))
        for file in glob.glob("*.csv"):
            print(file)
            df = pd.read_csv(os.path.join(os.path.join(user_input,folder),file),index_col=None,header=0)
            if not first:
                #df['label'] = 'benign'
                #df['malware_family'] = 'benign' # m.group(1)
                #df['file_name'] = file
                df.to_csv(save_file,index=False, header=False)
            else:
                first = False
                #df['label'] = 'benign'
                #df['malware_family'] = 'benign' #m.group(1)
                #df['file_name'] = file
                df.to_csv(save_file,index=False)
save_file.close()
