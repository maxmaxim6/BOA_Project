""" 
This class contains:
    1. Packet objects
    2. Methods which return information about a subset (or all) of the packets.
"""

class PacketContainer(object):
    
    # Class field - Packet container here
    
    def __init__(self):
        """ 
        self.list_of_packets
        or
        self.flows
        """
    
    
    """ Length in seconds """
    def duration(self):
        pass
    
    """ Total number of packets """
    def total_packets(self):
        pass
    
    """ Total number of packets with payload """
    def pl_total_packets(self):
        pass
    
    """ Total number of packets without payload """
    def no_pl_total_packets(self):
        pass
    
    """ Size of all packets in bytes """
    def size(self):
        pass
    
    """ Amount of packets """
    def __len__(self):
        pass
    
    """ Mean of packet size """
    def sizemean(self):
        pass
    
    """ Variance of packet size """
    def sizevar(self):
        pass
    
    """ Max packet size """
    def max_packet_size(self):
        pass
    
    """ Min packet size """
    def min_packet_size(self):
        pass
