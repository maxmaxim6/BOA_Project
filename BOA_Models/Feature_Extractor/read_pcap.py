
from asyncio.windows_events import NULL
from statistics import variance
from steelscript.wireshark.core.pcap import PcapFile
import pyshark
import numpy as np
import re
import subprocess
import datetime
import pandas as pd
from io import StringIO,BytesIO
import sys
import os

"""
Rename and reorder file
"""

"""
pdf = pcap.query(['frame.time_epoch', 'ip.src', 'ip.dst', 'ip.len', 'ip.proto'],
                starttime = pcap.starttime,
                duration='1min',
                as_dataframe=True)
"""

def gen_data_frame(path_str):
    pcap = PcapFile(path_str)
    #print '========='

    #print(repr(pcap.info()))
    #print '========='
    """
    frame.number,
    'frame.time_epoch',                                             
    'frame.time_delta', 
     tcp.ack,
    'frame.len', 
    'frame.cap_len', 
    'frame.marked', 
    ssl.handshake.session_id_length,
    ssl.handshake.comp_methods_length,
    tcp.options.wscale.shift,
    tcp.options.mss_val,
    'ip.src', 
    'ip.dst', 
    'ip.len', 
    'ip.flags', 
    'ip.flags.rb', 
    'ip.flags.df', 
    'ip.flags.mf', 
    'ip.frag_offset', 
    'ip.ttl', 
    'ip.proto',
    'tcp.srcport', 
    'tcp.dstport',
    udp.srcport,
    udp.dstport 
    'tcp.len', 
    'tcp.nxtseq', 
    'tcp.hdr_len', 
    'tcp.flags.cwr', 
    'tcp.flags.urg', 
    'tcp.flags.push', 
    'tcp.flags.syn' ,
    'tcp.window_size',
    'tcp.checksum'
    """
    #pcap.info()
    """

    frame.number,
    frame.time_epoch,
    frame.time_delta,
    frame.len,
    tcp.ack,
    frame.cap_len,
    frame.marked,
    ssl.handshake.session_id_length,
    ssl.handshake.comp_methods_length,
    tcp.options.wscale.shift,
    tcp.options.mss_val,
    ip.src,
    ip.dst,
    ip.len,
    ip.flags,
    ip.flags.rb,
    ip.flags.df,
    ip.flags.mf,
    ip.frag_offset,
    ip.ttl,
    ip.proto,
    ip.checksum_good,
    tcp.srcport,
    tcp.dstport,
    udp.srcport,
    udp.dstport,
    tcp.len,
    tcp.nxtseq,
    tcp.hdr_len,
    tcp.flags.cwr,
    tcp.flags.urg,
    tcp.flags.push,
    tcp.flags.syn,
    tcp.flags.ack,
    tcp.flags.reset,
    tcp.window_size,
    tcp.checksum,
    tcp.checksum_good,
    tcp.checksum_bad,
    tcp.analysis.keep_alive,
    ssl.record.version,
    ssl.handshake.type,
    ssl.handshake.cipher_suites_length,
    ssl.handshake.extensions_server_name,
    ssl.handshake.extension.type,
    isPeak,
    peak_num
    
    
    
    
    pdf = pcap.query([
	# 'frame.time_epoch',
	'frame.time_delta',
	# 'frame.pkt_len',
	# 'frame.len',
	# 'frame.cap_len',
	# 'frame.marked',
	'ip.src',
	'ip.dst',
	'ip.len',
	'ip.flags',
	# 'ip.flags.rb',
	# 'ip.flags.df',
	# 'ip.flags.mf',
	# 'ip.frag_offset', # Generates unexpected behaviour in steelscript-wireshark
	'ip.ttl',
	# 'ip.proto',
	# 'ip.checksum_good',
	'tcp.srcport',
	'tcp.dstport',
	'tcp.len',
	# 'tcp.nxtseq',
	# 'tcp.hdr_len',
	# 'tcp.flags.cwr',
	# 'tcp.flags.urg',
	# 'tcp.flags.push',
	# 'tcp.flags.syn',
	# 'tcp.window_size',
	# 'tcp.checksum',
	# 'tcp.checksum_good',
	# 'tcp.checksum_bad',
	# 'udp.length',
	# 'udp.checksum_coverage',
	# 'udp.checksum',
	# 'udp.checksum_good',
	# 'udp.checksum_bad'
    ],
    #starttime = pcap.starttime,
    as_dataframe=True)
    """
    
    pdf = pcap.query([
    'frame.time_delta',
	'ip.src',
	'ip.dst',
	'ip.len',
    'tcp.srcport',
	'tcp.dstport',
	'tcp.len',
	],
    starttime = pcap.starttime,
    as_dataframe=True)
    


    print ('=======')
    print ('pdf len: ' + repr(len(pdf)))


    return pdf


def read_pcap(filename, fields=[], display_filter="",
              timeseries=False, strict=False):
    """ Read PCAP file into Pandas DataFrame object.
    Uses tshark command-line tool from Wireshark.

    filename:       Name or full path of the PCAP file to read
    fields:         List of fields to include as columns
    display_filter: Additional filter to restrict frames
    strict:         Only include frames that contain all given fields
                    (Default: false)
    timeseries:     Create DatetimeIndex from frame.time_epoch
                    (Default: false)

    Syntax for fields and display_filter is specified in
    Wireshark's Display Filter Reference:

      http://www.wireshark.org/docs/dfref/
    """

    """
    pcap_path = '/home/jon/workspace/pcap-feature-extractor/data/L_cyber_chrome_09-17__11_38_11/L_cyber_chrome_09-17__11_38_11.pcap.TCP_10-0-0-14_35015_192-229-233-25_443.pcap'
    cmd = 'tshark -r %s -T fields -E occurrence=a -E aggregator=, -e frame.time_epoch -e frame.time_delta -e frame.len -e frame.cap_len -e frame.marked -e ip.src -e ip.dst -e ip.len -e ip.flags -e ip.flags.rb -e ip.flags.df -e ip.flags.mf -e ip.frag_offset -e ip.ttl -e ip.proto -e ip.checksum_good -e tcp.srcport -e tcp.dstport -e tcp.len -e tcp.nxtseq -e tcp.hdr_len -e tcp.flags.cwr -e tcp.flags.urg -e tcp.flags.push -e tcp.flags.syn -e tcp.window_size -e tcp.checksum -e tcp.checksum_good -e tcp.checksum_bad' % pcap_path
    table = subprocess.check_output(cmd.split())
    df = pd.read_table(StringIO(table), header=None, names=[ ... column names ... ])
    remove from cmd: -n : header=y : -R ''
    """
    
    if timeseries:
        fields = ["frame.time_epoch"] + fields
    fieldspec = " ".join("-e %s" % f for f in fields)
    """
    display_filters = fields if strict else []
    if display_filter:
        display_filters.append(display_filter)
    filterspec = "-R '%s'" % " and ".join(f for f in display_filters)
    """
    filterspec = ''
    #options = "-r %s -n -T fields -E header=y -E occurrence=a -E aggregator=, " % filename
    if len(fields) == 2:
        options1 = "-r %s -T fields -Y tls.handshake.type==1 -Y frame.time_epoch -E occurrence=a -E aggregator=, " % filename
        cmd = "tshark %s %s" % (options1, fieldspec)
    if len(fields) > 2:
        options2 = "-r %s -T fields -E occurrence=a -E aggregator=, " % filename
        cmd = "tshark %s %s" % (options2, fieldspec)
    


    if os.path.isfile(filename):

        try:

            table = subprocess.Popen(cmd.split(), shell=True, stdout=subprocess.PIPE ,stderr=subprocess.PIPE, universal_newlines=True)
            out, err = table.communicate()

        except subprocess.TimeoutExpired:
            table.kill()

            out, err = table.communicate()


        temp_split = out.split('\n')
        temp_sub_split = [item.split('\t') for item in temp_split]
        temp_sub_split.remove(temp_sub_split[len(temp_sub_split)-1])

        dict = {}
        i = 0
        for val in temp_sub_split:
            for f in fields:
                if i<len(fields) and f in dict:
                    dict[f].append(val[i])
                    i= i+1
                else:
                    dict[f] = [val[i]]
                    i= i+1
            i=0

        df = pd.DataFrame.from_dict(dict)
        return df
    
    else:
        df_empty = pd.DataFrame({'A' : []})
        return df_empty

    


""" Returns the upstream flow, downstream flow (in this order) from a given session DataFrame """
def gen_flows_up_down(pcap):


    dst_port = 0
    src_port = 0


    dst_port = int(pcap['tcp.dstport'].iloc[0])
    src_port = int(pcap['tcp.srcport'].iloc[0])

    if  dst_port == 443:


        ip_src = pcap['ip.src'].iloc[0]
        ip_dst = pcap['ip.dst'].iloc[0]

    elif src_port == 443:

        ip_src = pcap['ip.dst'].iloc[0]
        ip_dst = pcap['ip.src'].iloc[0]

    else:
        """ Throw exception? """
        print ('=====')
        print ('Port 443 not found')

        print ('=====')

    return pcap[pcap['ip.src']==ip_src], pcap[pcap['ip.src']==ip_dst]
