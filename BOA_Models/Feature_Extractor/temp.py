
from read_pcap import gen_data_frame, gen_flows_up_down, read_pcap
from general import gen_data_folders, space_to_underscore
from Flow import Flow
from Session import Session
from Converter import Converter
import pandas as pd
import numpy as np
import os
from os.path import join
import subprocess
from main import work, start_here

np.set_printoptions(precision=2, suppress=True)

"""
pcap1_path = '/home/jon/workspace/pcap-feature-extractor/data/L_cyber_chrome_09-17__11_38_11/L_cyber_chrome_09-17__11_38_11.pcap.TCP_10-0-0-14_33521_212-179-154-238_443.pcap'
pcap2_path = '/home/jon/workspace/pcap-feature-extractor/data/L_cyber_chrome_09-17__11_38_11/L_cyber_chrome_09-17__11_38_11.pcap.TCP_10-0-0-14_34965_192-229-233-25_443.pcap'

p1 = gen_data_frame(pcap2_path)
#p2 = gen_data_frame(pcap2_path)

fp1, fp2 = gen_flows(p1)

#print fp1[['ip.src','ip.dst']]
#print
#print fp2[['ip.src','ip.dst']]

f1 = Flow(fp1)
f2 = Flow(fp2)


""" """
sample = pcap_to_feature_vector(pcap2_path, ['packet_count', 'sizemean', 'sizevar'],1)


print len(f1)
print len(f2)
print 'min packet size'
print f1.min_packet_size()
print f2.min_packet_size()
print 'mean'
print f1.sizemean()
print f2.sizemean()
print 'variance'
print f1.sizevar()
print f2.sizevar()
print 'sum bytes'
print f1.size()
print f2.size()
print '--------'
print repr(sample)
"""

#sessions_to_samples('/home/jon/workspace/pcap-feature-extractor/data/L_cyber_chrome_09-17__11_38_11')

#gen_data_folders('/home/jon/workspace/pcap-feature-extractor/data')

sstr = os.getcwd()
sstr = sstr + '/data'
# gen_data_folders(sstr)

# main_script(ROOT_DIRECTORY=sstr,
#             output_filename='20_12_15_samples.csv',
#             rename_space_underscore=False,
#             feature_list=['packet_count', 'sizemean', 'sizevar', 'std_fiat', 'std_biat', 'fpackets', 'bpackets', 'fbytes', 'bbytes', 'min_fiat', 'min_biat', 'max_fiat', 'max_biat', 'std_fiat', 'std_biat', 'mean_fiat', 'mean_biat', 'min_fpkt', 'min_bpkt', 'max_fpkt', 'max_bpkt', 'std_fpkt', 'std_bpkt', 'mean_fpkt', 'mean_bpkt'])

# print repr(sdf)

start_here()


# print repr(conv.feature_methods)
# for sample in conv:
#     print sample
#     print

# print '======================='
# print 'marker'
# cmd = 'tshark -r /home/jon/workspace/pcap-feature-extractor/data/L_cyber_chrome_09-17__11_38_11/L_cyber_chrome_09-17__11_38_11.pcap.TCP_10-0-0-14_33521_212-179-154-238_443.pcap -T fields -E occurrence=a -E aggregator=, -e frame.time_delta -e ip.src -e ip.dst -e ip.len -e ip.flags -e ip.ttl -e tcp.srcport -e tcp.dstport -e tcp.len -e tcp.checksum'
# out = subprocess.check_output(cmd.split())
# out2 = subprocess.Popen(cmd, shell = True, stdout=subprocess.PIPE)
# df = pd.read_table(out)
# print repr(df)
# print repr(out2.stdout)
# fields = ['frame.time_epoch', 'frame.time_delta', 'frame.len', 'frame.cap_len', 'frame.marked', 'ip.src', 'ip.dst', 'ip.len', 'ip.flags', 'ip.flags.rb', 'ip.flags.df', 'ip.flags.mf', 'ip.frag_offset', 'ip.ttl', 'ip.proto', 'ip.checksum_good', 'tcp.srcport', 'tcp.dstport', 'tcp.len', 'tcp.nxtseq', 'tcp.hdr_len', 'tcp.flags.cwr', 'tcp.flags.urg', 'tcp.flags.push', 'tcp.flags.syn' ,'tcp.window_size','tcp.checksum','tcp.checksum_good', 'tcp.checksum_bad']
# df = read_pcap('/home/jon/workspace/pcap-feature-extractor/data/L_cyber_chrome_09-17__11_38_11/L_cyber_chrome_09-17__11_38_11.pcap.TCP_10-0-0-14_33521_212-179-154-238_443.pcap', fields=fields)
# print repr(df)
print ('=======================')
print ('marker')
