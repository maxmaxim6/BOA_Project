
from hcl_helpers import read_label_data
from general import is_pcap_session, space_to_underscore, write_pcap_filenames
from os.path import join
import os

# file_path = '/home/jon/workspace/pcap-feature-extractor/data/L_cyber_chrome_09-17__11_38_11/label_data.hcl'
# os, browser, application, service = read_label_data(file_path)
# print 'os: ' + os + ' browser: ' + browser + ' application: ' + application + ' service: ' + service


cwd = os.getcwd()
filename = join(cwd,'filenames.csv')
write_pcap_filenames(['a.pcap', 'b.pcap', 'c.pcap'], filename)
