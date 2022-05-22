
from Flow import Flow
from Session import Session
import unittest
import pandas as pd
import numpy as np

"""
FIX:
The data is recreated for each test
"""

class TestSession1(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
			pass

	def setUp(self):
		columns = [
		'frame.time_epoch',
		'frame.time_delta',
		'frame.len',
		'frame.cap_len',
		'frame.marked',
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
		'ip.checksum_good',
		'tcp.srcport',
		'tcp.dstport',
		'tcp.len',
		'tcp.nxtseq',
		'tcp.hdr_len',
		'tcp.flags.cwr',
		'tcp.flags.urg',
		'tcp.flags.push',
		'tcp.flags.syn',
		'tcp.window_size',
		'tcp.checksum',
		'tcp.checksum_good',
		'tcp.checksum_bad',
		]
		self.df1 = pd.DataFrame(np.zeros((5,29)), columns=columns)
		self.df2 = pd.DataFrame(np.zeros((4,29)), columns=columns)
		self.df1['frame.len'] = [10,100,60,40,200]
		self.df2['frame.len'] = [30,70,300,100]
		self.df1['ip.src'] = ['10.1.0.2','10.1.0.2','10.1.0.2','10.1.0.2','10.1.0.2']
		self.df1['ip.dst'] = ['8.8.8.8','8.8.8.8','8.8.8.8','8.8.8.8','8.8.8.8']
		self.df1['tcp.srcport'] = ['2222','2222','2222','2222','2222']
		self.df1['tcp.dstport'] = ['443','443','443','443','443']
		self.df2['ip.src'] = ['8.8.8.8','8.8.8.8','8.8.8.8','8.8.8.8']
		self.df2['ip.dst'] = ['10.1.0.2','10.1.0.2','10.1.0.2','10.1.0.2']
		self.df2['tcp.srcport'] = ['443','443','443','443']
		self.df2['tcp.dstport'] = ['2222','2222','2222','2222']
		self.sess_frame = pd.concat((self.df1, self.df2), ignore_index=True)
		
		self.s = Session(self.sess_frame)
		

	def test_len(self):
		self.assertEqual(len(self.s), 9)

	def test_mean(self):
		print("!!! ",self.s.mean_packet_size())
		self.assertAlmostEqual(self.s.mean_packet_size(), 101.11111111)

	def test_variance(self):
		self.assertAlmostEqual(self.s.sizevar(), 8637.11111111)

if __name__ == '__main__':
	unittest.main()
