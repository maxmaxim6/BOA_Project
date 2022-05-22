
# from feature_extraction.flowpic import FlowPic2019
from feature_extraction.protocol_header_fields import ProtocolHeaderFields
from feature_extraction.n_bytes import NBytes
from feature_extraction.stnn import STNN
from feature_extraction.n_bytes_per_packet import  NBytesPerPacket
from nfstream import NFStreamer  # https://www.nfstream.org/docs/api

def extract_features(input_path,output_path):
    
    plugins = [
        NBytes(),
        ProtocolHeaderFields(),
        STNN(),
        NBytesPerPacket(n=100,max_packets=2)
    ]
    
    my_streamer = NFStreamer(source=input_path,
                                    decode_tunnels=True,
                                    bpf_filter="udp or tcp",
                               
                                    promiscuous_mode=True,
                                    snapshot_length=1536,
                                    idle_timeout=99999999999999,
                                    active_timeout=999999999999999,
                                    accounting_mode=3,
                                    udps=plugins,
                                    n_dissections=20,
                                    statistical_analysis=True,
                                    splt_analysis=0,
                                    n_meters=1,
                                    performance_report=0)
    df = my_streamer.to_pandas()
    df = df[df['bidirectional_bytes'] >= 784]
    df.to_csv(output_path)

    print('Exiting...')
