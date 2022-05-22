
from PacketContainer import PacketContainer
from read_pcap import gen_data_frame, gen_flows_up_down, read_pcap
from Flow import Flow
import pandas as pd
from statistics import variance

"""
FIX:
"""

"""
Class fields:
sess - Session DataFrame
"""

class Session(PacketContainer):

    def __init__(self, s):
        self.sess = s
        self.flow_up, self.flow_down = gen_flows_up_down(self.sess)
        self.flow_up, self.flow_down = Flow(self.flow_up), Flow(self.flow_down)


    """ Whats the difference between this function and the ctor? """
    @classmethod
    def from_filename(cls, path_str, fields=['frame.time_epoch', 'frame.time_delta', 'frame.len', 'frame.cap_len', 'frame.marked', 'ip.src', 'ip.dst', 'ip.len', 'ip.flags', 'ip.flags.rb', 'ip.flags.df', 'ip.flags.mf', 'ip.frag_offset', 'ip.ttl', 'ip.proto', 'tcp.srcport', 'tcp.dstport', 'tcp.len', 'tcp.nxtseq', 'tcp.hdr_len', 'tcp.flags.cwr', 'tcp.flags.urg', 'tcp.flags.push', 'tcp.flags.syn' ,'tcp.window_size']):
        #sess = gen_data_frame(path_str)
        
        sess = read_pcap(path_str,fields=fields)
        return cls(sess)

    """ Length in seconds """
    def duration(self):
        pass

    """ Total number of packets with payload """
    def pl_total_packets(self):
        pass

    """ Total number of packets without payload """
    def no_pl_total_packets(self):
        pass

    """ Size of all packets in bytes """
    def size(self):
        return self.flow_up.size() + self.flow_down.size()

    """ Amount of packets """
    def __len__(self):
        return len(self.sess)

    """ Total number of packets """
    def packet_count(self):
        return len(self)

    """ Mean of packet size """
    def mean_packet_size(self):
        #print("Session mean: ",self.sess['frame.len'])
        self.sess['frame.len'] = pd.to_numeric(self.sess['frame.len'],downcast="float")
        return self.sess['frame.len'].mean()

    """ Variance of packet size """
    def sizevar(self):
        self.sess['frame.len'] = pd.to_numeric(self.sess['frame.len'],downcast="float")
        return self.sess['frame.len'].var()

    """ Max packet size """
    def max_packet_size(self):
        ####new
        self.sess['frame.len'] = pd.to_numeric(self.sess['frame.len'],downcast="float")
        return self.sess['frame.len'].max()

    """ Min packet size """
    def min_packet_size(self):
        ###new
        self.sess['frame.len'] = pd.to_numeric(self.sess['frame.len'],downcast="float")
        return self.sess['frame.len'].min()

    """ # Packets in forward direction (fpackets) """
    def fpackets(self):
        return len(self.flow_up)

    """ # Packets in backward direction (bpackets) """
    def bpackets(self):
        return len(self.flow_down)

    """ # Bytes in forward direction (fbytes) """
    def fbytes(self):
        return self.flow_up.size()

    """ # Bytes in backward direction (bbytes) """
    def bbytes(self):
        return self.flow_down.size()

    """ Min forward inter-arrival time (min_fiat) """
    def min_fiat(self):
        return self.flow_up.min_time_delta()

    """ Min backward inter-arrival time (min_biat) """
    def min_biat(self):
        return self.flow_down.min_time_delta()

    """ Max forward inter-arrival time (max_fiat) """
    def max_fiat(self):
        return self.flow_up.max_time_delta()

    """ Max backward inter-arrival time (max_biat) """
    def max_biat(self):
        return self.flow_down.max_time_delta()

    """ Standard deviation of forward inter- arrival times (std_fiat) """
    def std_fiat(self):
        return self.flow_up.std_time_delta()

    """ Standard deviation of backward inter- arrival times (std_biat) """
    def std_biat(self):
        return self.flow_down.std_time_delta()

    """ Mean forward inter-arrival time (mean_fiat) """
    def mean_fiat(self):
        return self.flow_up.mean_time_delta()

    """ Mean backward inter-arrival time (mean_biat) """
    def mean_biat(self):
        return self.flow_down.mean_time_delta()

    """ Min forward packet length (min_fpkt) """
    def min_fpkt(self):
        return self.flow_up.min_packet_size()

    """ Min backward packet length (min_bpkt) """
    def min_bpkt(self):
        return self.flow_down.min_packet_size()

    """ Max forward packet length (max_fpkt) """
    def max_fpkt(self):
        return self.flow_up.max_packet_size()

    """ Max backward packet length (max_bpkt) """
    def max_bpkt(self):
        return self.flow_down.max_packet_size()

    """ Std deviation of forward packet length (std_fpkt) """
    def std_fpkt(self):
        return self.flow_up.std_packet_size()

    """ Std deviation of backward packet length (std_bpkt) """
    def std_bpkt(self):
        return self.flow_down.std_packet_size()

    """ Mean forward packet length (mean_fpkt)	"""
    def mean_fpkt(self):
        return self.flow_up.mean_packet_size()

    """ Mean backward packet length (mean_bpkt) """
    def mean_bpkt(self):
        return self.flow_down.mean_packet_size()
