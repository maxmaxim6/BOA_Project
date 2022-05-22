

import unittest
from general import is_pcap_session

class TestGeneral(unittest.TestCase):

	def setUp(self):
		self.pcap1_path = '/home/jon/workspace/pcap-feature-extractor/data/L_cyber_chrome_09-17__11_38_11/L_cyber_chrome_09-17__11_38_11.pcap.TCP_10-0-0-14_33521_212-179-154-238_443.pcap'
		self.pcap2_path = '/home/jon/workspace/pcap-feature-extractor/data/t.pcap'

	def test_is_pcap_session_True(self):
		self.assertTrue(is_pcap_session(self.pcap1_path))

	def test_is_pcap_session_False(self):
		self.assertFalse(is_pcap_session(self.pcap2_path))

if __name__ == '__main__':
	unittest.main()
