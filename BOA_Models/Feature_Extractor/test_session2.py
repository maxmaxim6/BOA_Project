

from Flow import Flow
from Session import Session
import unittest
import pandas as pd
import os
import subprocess

"""
FIX:
The data is recreated for each test
"""

class TestSession2(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		#sstr = os.getcwd()
		sstr = 'C:\Users\TAL\AppData\Local\Programs\Python\Python39\robust_project\pcap-feature-extractor-master\data\L_cyber_chrome_09-17__11_38_11\L_cyber_chrome_09-17__11_38_11.pcap.TCP_10-0-0-14_33521_212-179-154-238_443.pcap'
		# sstr = sstr + '/data/L_cyber_chrome_09-17__11_38_11/L_cyber_chrome_09-17__11_38_11.pcap.TCP_10-0-0-14_34965_192-229-233-25_443.pcap'
		cls.s = Session.from_filename(sstr)
		cmd = 'tshark -r C:\Users\TAL\AppData\Local\Programs\Python\Python39\robust_project\pcap-feature-extractor-master\data\L_cyber_chrome_09-17__11_38_11\L_cyber_chrome_09-17__11_38_11.pcap.TCP_10-0-0-14_33521_212-179-154-238_443.pcap -T fields -E occurrence=a -E aggregator=, -e frame.time_epoch -e frame.time_delta -e frame.len -e frame.cap_len -e frame.marked -e ip.src -e ip.dst -e ip.len -e ip.flags -e ip.flags.rb -e ip.flags.df -e ip.flags.mf -e ip.frag_offset -e ip.ttl -e ip.proto -e tcp.srcport -e tcp.dstport -e tcp.len -e tcp.nxtseq -e tcp.hdr_len -e tcp.flags.cwr -e tcp.flags.urg -e tcp.flags.push -e tcp.flags.syn -e tcp.window_size -e tcp.checksum'
		cls.table = subprocess.check_output(cmd.split()).splitlines()


	"""
	def setUp(self):
		self.s = Session(self.sess_frame)
	"""

	def test_len(self):
		print ('====')
		print ('tshark output lines: ') + repr(len(self.table))
		print ('====')
		for i in range(1000):
			self.assertEqual(len(self.s), 5890)
			# self.assertEqual(len(self.s), 196)
	"""
	def test_mean(self):
		self.assertAlmostEqual(self.s.sizemean(), 101.11111111)

	def test_variance(self):
		self.assertAlmostEqual(self.s.sizevar(), 8636.11111111)
	"""

if __name__ == '__main__':
	unittest.main()
