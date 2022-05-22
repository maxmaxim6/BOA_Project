""" 
This class contains:
    1. A Packet object from scapy.
    2. Methods which return information about the packet.
"""

class Packet(object):
    
  
    def __init__(self, p):
        self.scapy_packet = p
    
    
    """ Is the packet upstream of downstream """
    def packet_direction(self):
        pass
    
    
    """ Delta time from previous """
    def frame_time_delta(self):
        pass
    
    
    """ Frame Length """
    def frame_len(self):
        pass
    
    
    """ Capture Length """
    def frame_cap_len(self):
        pass
    
    
    """ Frame is marked """
    def frame_marked(self):
        pass
    
    
    """ IP Header length """
    def ip_len(self):
        pass 
    
    
    """ IP Flags """
    def ip_flags(self):
        pass
    
    
    """ IP Flags: Reserved bit """
    def ip_flags_rb(self):
        pass
    
    
    """ IP Flags dont fragment """
    def ip_flags_df(self):
        pass 
    
    
    """ IP Flags: More fragments """
    def ip_flags_mf(self):
        pass
    
    
    """ IP Fragment offset """
    def ip_frag_offset(self):
        pass
    
    
    """ IP 	Time to live """
    def ip_ttl(self):
        pass
    
    
    """ IP Protocol """
    def ip_proto(self):
        pass
    
    
    """ IP Header checksum is set to true """
    def ip_checksum_good(self):
        pass
    
    
    """ TCP Segment Length """
    def tcp_len(self):
        return self.scapy_packet['IP'].len
  
  
    """ TCP Next sequence number """ 
    def tcp_nxtseq(self):
        pass
    
    
    """ TCP Header length """
    def tcp_hdr_len(self):
        pass
    
    
    """ TCP Flags: Congestion Window """
    def tcp_flags_cwr(self):
        pass


    """ TCP Flags: Urgent """
    def tcp_flags_urg(self):
        pass
    
    
    """ TCP Flags: Push """
    def tcp_flags_push(self):
        pass
    
    
    """ TCP Flags: Syn """
    def tcp_flags_syn(self):
        pass
    
    
    """ TCP Window size """
    def tcp_window_size(self):
        pass
    
    
    """ TCP Checksum """
    def tcp_checksum(self):
        pass
    
    
    """ TCP Checksum is set to true (1) """
    def tcp_checksum_good(self):
        pass
    
    
    """ TCP checksum is set to false (0) """
    def tcp_checksum_bad(self):
        pass
	

	
    """ UDP Length """
    def udp_length(self):
        pass
    
    
    """ UDP Checksum coverage """
    def udp_checksum_coverage(self):
        pass
    
    
    """ UDP Checksum """
    def udp_checksum(self):
        pass
    
    
    """ UDP Checksum is set to true (1) """   
    def udp_checksum_good(self):
        pass
    
    
    """ UDP Checksum is set to false (0) """
    def udp_checksum_bad(self):
        pass
