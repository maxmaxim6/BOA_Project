from asyncio.windows_events import NULL
from cmath import nan
import os
import path
from glob import glob
from os import path
from os import listdir
from os.path import isfile, join
from read_pcap import read_pcap
import csv

def cleanup_pyc(DIRECTORY):
    d = path(DIRECTORY)
    files = d.walkfiles("*.pyc")
    for file in files:
        file.remove()
        print ("Removed {} file".format(file))


"""
Assuming a relevant pcap directory contains a .hcl file with label details.
This allows a non strict folder hierarchy i.e.
data/
    any_folder_order/
        relevant_folder1/
            label_data.hcl
            *.pcap
    dummy_folder_name/
        relevant_folder2/
            label_data.hcl
            *.pcap

-----------------
Currently assuming that if a single pcap file in a directory is a session
pcap, all other pcap files in the directory are also session pcaps.
Therefor the if clause checks if any of the pcap files in a given
directory is a session pcap. If true the directory is added to the list
of relevant directories.
"""
def gen_data_folders(PARENT_DIRECTORY):
    #root = PARENT_DIRECTORY
    d = os.path.abspath(PARENT_DIRECTORY)
    #print("abs_path: ",d)
    l = []
    
    for root, dirs, files in os.walk(d):
        print(root," who is root")
        #print(files," who is files")
        #for filename in files:
            #print(filename,"filename")
        #    filename = os.path.join(root, filename)
        #    print(filename,"filename")
        l.append(root)

            

    """
    for root, dier ,files in os.walk(d):
        
        print(root," who is root")
        print(files," who is files")
        # if any(file.endswith('.hcl') for file in files) and any(is_pcap_session(file) for file in files):
 
        if any(is_pcap_session(os.path.join(root,file)) for file in files):
            l.append(os.path.abspath(root))
    """
    l = l[1:]
    print("After: ",l)
    return l

""" Returns a list of pcap file names from a given folder """
def gen_pcap_filenames(folder_name):
        # return [join(folder_name, f) for f in listdir(folder_name) if (isfile(join(folder_name, f)) and ('hcl' not in f) and ('pcap' in f)) ]
        file_names = [join(folder_name, f) for f in listdir(folder_name) if (isfile(join(folder_name, f)) and is_tcp_prot(f)) ]
        
        # print file_names
        return file_names


"""
Write a list of pcap file names to a given filename
DOES NOT WORK YET
"""
def write_pcap_filenames(filename_list, file_name):
        with open(file_name, "wb") as f:
            writer = csv.writer(f)
            writer.writerow(filename_list)


"""
Labels per combination:
    os = { Linux, Windows, OSX }
    browser = { Chrome, FireFox, IExplorer }
    application = { , }
    service = { , }

    0 = (Linux, Chrome)
    1 = (Linux, FireFox)
    2 = (Windows, Chrome)
    3 = (Windows, FireFox)
    4 = (Windows, IExplorer)
    5 = (OSX, Safari)

"""
def gen_label_app_os_browser(os, browser, application):
    
    if os == 'Linux':
        if browser == 'Chrome':
            if application == 'unknown':
                return -1
            elif application == 'dropbox':
                return 0
            elif application == 'google':
                return 1
            elif application == 'microsoft':
                return 2
            elif application == 'twitter':
                return 3
            elif application == 'vine':
                return 4
            elif application == 'youtube':
                return 5
        if browser == 'FireFox':
            if application == 'unknown':
                return -2
            elif application == 'dropbox':
                return 6
            elif application == 'google':
                return 7
            elif application == 'microsoft':
                return 8
            elif application == 'twitter':
                return 9
            elif application == 'vine':
                return 10
            elif application == 'youtube':
                return 11 
    
    if os == 'Windows':
        if browser == 'Chrome':
            if application == 'unknown':
                return -3
            elif application == 'dropbox':
                return 12
            elif application == 'google':
                return 13
            elif application == 'microsoft':
                return 14
            elif application == 'twitter':
                return 15
            elif application == 'vine':
                return 16
            elif application == 'youtube':
                return 17
        if browser == 'FireFox':
            if application == 'unknown':
                return -4
            elif application == 'dropbox':
                return 18
            elif application == 'google':
                return 19
            elif application == 'microsoft':
                return 20
            elif application == 'twitter':
                return 21
            elif application == 'vine':
                return 22
            elif application == 'youtube':
                return 23
        if browser == 'IExplorer':
            if application == 'unknown':
                return -4
            elif application == 'dropbox':
                return 24
            elif application == 'google':
                return 25
            elif application == 'microsoft':
                return 26
            elif application == 'twitter':
                return 27
            elif application == 'vine':
                return 28
            elif application == 'youtube':
                return 29
    if os == 'OSX':
        if browser == 'Safari':
            if application == 'unknown':
                return -5
            elif application == 'dropbox':
                return 30
            elif application == 'google':
                return 31
            elif application == 'microsoft':
                return 32
            elif application == 'twitter':
                return 33
            elif application == 'vine':
                return 34
            elif application == 'youtube':
                return 35

def gen_label_os(os, browser, application, service):
    
    if os == 'Linux':
        return 0
    elif os == 'Windows':
        return 1
    elif os == 'OSX':
        return 2
    
def gen_label_browser(os, browser, application, service):
    """
    if os == 'Linux':
        if browser == 'Chrome':
            return 0
        elif browser == 'FireFox':
            return 1
    elif os == 'Windows':
        if browser == 'Chrome':
            return 2
        elif browser == 'FireFox':
            return 3
        elif browser == 'IExplorer':
            return 4
    elif os == 'OSX':
        if browser == 'Safari':
            return 5
    """
    
    if browser == 'Chrome':
        return 0
    elif browser == 'FireFox':
        return 1
    elif browser == 'IExplorer':
        return 2
    elif browser == 'Safari':
        return 3



"""
Parse a folder name and return the os + browser
Currently returns os only.
Assumes the following format:
L_cyber_chrome_09-17__11_38_11
"""
def extract_app_from_dict(dict_from_csv,sv_name):

    if sv_name in dict_from_csv:
        return dict_from_csv[sv_name]
    else :
        return None


def parse_app_name(app_name):

    if app_name == 'dropbox':
        return 0
    elif app_name == 'facebook':
        return 1
    elif app_name == 'google':
        return 2
    elif app_name == 'youtube':
        return 3
    elif app_name == 'microsoft':
        return 4
    elif app_name == 'twitter':
        return 5
    elif app_name == 'vine':
        return 6
    elif app_name == 'whatsapp':
        return 7
    elif app_name == 'wireshark':
        return 8
    elif app_name == 'imgur':
        return 9
    elif app_name == 'unknown':
        return -1

    '''
    CLASS -1 = unknown
    CLASS 0 = dropbox
    CLASS 2 = google
    CLASS 4 = microsoft 
    CLASS 5 = twitter
    CLASS 6 = vine
    CLASS 9 = youtube
    '''

def parse_folder_name_os(folder_name):
    temp = folder_name.split(os.sep)
    print("before split: ",temp)
    temp.reverse()
    print("After revers: ",temp)
    tokens = temp[0].split('_')

    """
    dropbox
    facebook
    google
    imgur
    microsoft
    soundcloud
    twitter
    --unknown
    vine
    whatsapp
    wireshark
    youtube
    """

    
    if tokens[0] == 'L' or tokens[0] == 'l':
        return 'Linux'
    elif tokens[0] == 'W' or tokens[0] == 'w':
        return 'Windows'
    elif tokens[0] == 'd' or tokens[0] == 'D':
        return 'OSX'
    
def parse_folder_name_browser(folder_name):
    temp = folder_name.split(os.sep)
    print("before split: ",temp)
    temp.reverse()
    print("After revers: ",temp)
    tokens = temp[0].split('_')
    
    if tokens[2] == 'chrome':
        return 'Chrome'
    elif tokens[2] == 'ff':
        return 'Firefox'
    elif tokens[2] == 'ie':
        return 'IExplorer'
    elif tokens[2] == 'safari':
        return 'Safari'
    
""" Return True if the given pcap is a session """
def is_tcp_prot(pcap_file):
    
    print("pcap_file: ",pcap_file)
    tcp ='.TCP_'
    ip_dst ='_443_'
    port_dst='_443.pcap'
    if tcp in pcap_file and (ip_dst in pcap_file or port_dst in pcap_file):
        print("True")
        return True
    return False
    

def is_pcap_session(pcap_path):
    print("pcap_path: ",pcap_path)
    if pcap_path.endswith('.pcap'):
        
        df = read_pcap(pcap_path, fields=['frame.time_epoch','tls.handshake.extensions_server_name'])
        if not df.empty:
            #print("df ",df)
            #print("app: ",df['tls.handshake.extensions_server_name'])
            for server in df['tls.handshake.extensions_server_name']:
                if len(server) > 1:
                    print("True")
                    return True
            #sni_count = len(df['frame.time_epoch'])
            #print("sni_count: ",sni_count)
            #if sni_count > 0:
                
            #    return True
            #else:

    return False

""" Replace space with underscore for all folder and file names """
def space_to_underscore(ROOT_FOLDER):
        d = path(ROOT_FOLDER)


        for root, dirs, files in os.walk(d):
            # print 'In ' + repr(str(root))
            # print '================='

            for filename in os.listdir(root): # parse through file list in the current directory
                # print 'Filename: ' + repr(str(filename))
                # print '================='

            	if filename.find(" ") > 0: # if an underscore is found
                    newfilename = filename.replace(' ','_')
                    # print 'newfilename: ' + repr(str(newfilename))

                    os.rename(join(root, filename), join(root, newfilename))
