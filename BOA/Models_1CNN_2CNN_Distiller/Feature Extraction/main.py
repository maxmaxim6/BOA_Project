from glob import glob
import os
import pandas as pd

from datetime import datetime
from feature_extraction_function import extract_features 
from multiprocessing import Process, freeze_support 
import torch.multiprocessing as mp

def feautre_extraction_from_dir(input_path,out_path):
    mp.freeze_support()

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    os.chdir(input_path)
    for file in glob("*.pcap"):
            print(file)
            pre,ext = os.path.splitext(file)
            if os.path.exists(os.path.join(out_path,pre+'.csv')):
                continue
            try:
                extract_features(os.path.join(input_path,file),os.path.join(out_path,pre+'.csv'))
            except Exception as e:
                    error_file = open('error_log.txt', 'a')
                    t = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                    error_file.write(str(t)+' '+str(file)+' Failed to NFstream: '+str(e)+'\n') 
                    error_file.close() 
                    print('Failed to read ',error_file,'because of a given error with the nfstream.', e )




from multiprocessing import Process, freeze_support
if __name__ == '__main__':
    # freeze_support() 
    # Process(target=f).start()
    # extractor = parallelTestModule.ParallelExtractor()
    # extractor.runInParallel(numProcesses=2, numThreads=4)

    
    input_path = r'C:\Users\maxma\Desktop\PROJECT\mapG\mapp_graph\splitcapfiles'
    out_path = r'C:\Users\maxma\Desktop\PROJECT\mapG\mapp_graph\splitcapfiles'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    os.chdir(input_path)

    for folder in glob("*/")+[input_path]:
            print(folder)
            feautre_extraction_from_dir(os.path.join(input_path,folder),os.path.join(out_path,folder))