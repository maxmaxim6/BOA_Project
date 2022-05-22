from asyncio.windows_events import NULL
from queue import Empty
import subprocess
from Converter import Converter
from general import space_to_underscore
from read_pcap import read_pcap
import os
import csv
import ast
import pandas as pd
import subprocess
"""
Convert all relevant pcap files in the given ROOT_DIRECTORY with the given feature_list and save the results to the output_filename
"""

def work(
    ROOT_DIRECTORY,
    output_filename='samples.csv',
    rename_space_underscore=False,
    feature_list=['packet_count', 'mean_packet_size', 'sizevar', 'std_fiat', 'std_biat', 'fpackets', 'bpackets', 'fbytes', 'bbytes', 'min_fiat', 'min_biat', 'max_fiat', 'max_biat', 'std_fiat', 'std_biat', 'mean_fiat', 'mean_biat', 'min_fpkt', 'min_bpkt', 'max_fpkt', 'max_bpkt', 'std_fpkt', 'std_bpkt', 'mean_fpkt', 'mean_bpkt']
    ):

    if rename_space_underscore:
        space_to_underscore(ROOT_DIRECTORY)

    features = feature_list
    conv = Converter(ROOT_DIRECTORY, feature_list)
    conv.activate(features)
    print("feature_list : ",feature_list)
    print("first : ",feature_list[0])
    print("second : ",feature_list[1])
    feature_list.append('label')
    print("my_path: ",ROOT_DIRECTORY + '\\' + output_filename)
    conv.write_to_csv(ROOT_DIRECTORY + '\\' + output_filename, column_names=feature_list)
    #conv.write_to_csv_tab(ROOT_DIRECTORY + '\\' + output_filename_tab, separator='\t', column_names=feature_list)




def app_os_browser_labels_from_csv():
    
    with open('all_labels.csv', mode='r') as inp:
        reader = csv.reader(inp)
        dict_from_csv = {(rows[1],rows[2],rows[3]):rows[4] for rows in reader}
    return dict_from_csv

def my_dict():
    
    with open('all_ids.csv', mode='r') as inp:
        reader = csv.reader(inp)
        mydict = {(rows[1],rows[2]): rows[4] for rows in reader}
    #print("mydict: ",mydict)
    return mydict

def server_name(ROOT_DIRECTORY):
    d = os.path.abspath(ROOT_DIRECTORY)
    
    mydict = {}
    cnt = 0
    with open('server_name.csv', mode='r') as inp:
        reader = csv.reader(inp)
        dict_from_csv = {rows[0]:rows[1] for rows in reader}
    

    
    for root, dier ,files in os.walk(d):
        cnt = cnt+1
        if cnt > 1:
            mydict[root] = []

        print("server_name root: ",root)
        l = []
        
        for file in files:
            server_name = ''
            pcap =  os.path.join(root,file) 
            if pcap.endswith('.pcap'):
                df = read_pcap(pcap, fields=['frame.time_epoch','tls.handshake.extensions_server_name'])
                #print(df['tls.handshake.extensions_server_name'])
                if not df.empty:
                    for server in df['tls.handshake.extensions_server_name']:
                        if len(server)>1: 
                            server_name = server
                
                if server_name != None:
                    l.append((pcap,server_name))
        if len(l) > 0 and cnt > 1:
            mydict[root] = l
        
        
    #print()W_Pc_ie_10-15__21_45_06.pcap.TCP_10-0-0-18_51860_104-244-42-193_443.pcap.Seconds_1444974000.pcap.Seconds_1444974379.pcap
    #print(mydict)
    return dict_from_csv,mydict
    
def my_split_cap():

    print ("Enter data root directory: ")
    PARENT_DIRECTORY = input()
    d = os.path.abspath(PARENT_DIRECTORY)
    #print("abs_path: ",d)
    l = []
    #spcap_exe = 
    for root, dirs, files in os.walk(d):
        root_name = root.split('\\')
        print(root," who is root")
        print(root_name," who is root_name")
        for filename in files:
            #print(filename,"filename")
            filename = os.path.join(root, filename)
            new_directory = "F:\\splitcapfiles" +'\\'+ root_name[len(root_name)-1]
            cmd = "C:\\Users\\TAL\\SplitCap.exe -r %s -s session -o %s" % (filename,new_directory)
            print("who is cmd :",cmd)
            print("who is cmd after split :",cmd.split())
            subprocess.run(cmd.split())
            
            print(filename,"filename")
            #l.append(filename)

def my_split_cap1():

    print ("Enter data root directory: ")
    PARENT_DIRECTORY = input()
    d = os.path.abspath(PARENT_DIRECTORY)
    #print("abs_path: ",d)
    for root, dirs, files in os.walk(d):
        print(root," who is root")
        for filename in files:
            filename = os.path.join(root, filename)
            cmd ="tshark -q -z conv,tcp -r %s" % filename
            table = subprocess.Popen(cmd.split(), shell=True, stdout=subprocess.PIPE ,stderr=subprocess.PIPE, universal_newlines=True)
            out, err = table.communicate()
            #print("out: ",out)
            print()
            #print("out split: ",out.split('\n'))
            out_split = out.split('\n')
            sub = out_split[5:len(out_split)-2]
            print("sub: ")
            print(sub)
            if len(sub)>1:
                print("true")
            else:
                print("false")
            print()


def start_here():
    print ("Assuming config file is up-to-date")
    print ('---')
    print ("Enter data root directory: ")
    ROOT_DIR = input()
    print ("The system does not cope with spaces in folder / file names.")
    print ("Replace spaces with underscores in given directory?")
    print ("Type y / n")
    rename_space_under_input = input()
    #C:\Users\TAL\Desktop\data_raw
    #C:\Users\TAL\AppData\Local\Programs\Python\Python39\robust_project\pcap-feature-extractor-master\my_data\my_test
    if rename_space_under_input == 'y':
        rename_space_under = True
    elif rename_space_under_input == 'n':
        rename_space_under = False




    out_file ='samples.csv'



    work(ROOT_DIRECTORY=ROOT_DIR, output_filename=out_file, rename_space_underscore=rename_space_under)
   
if __name__ == '__main__':
    start_here()
   