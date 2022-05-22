

from Flow import Flow
from Session import Session
import unittest
import pandas as pd


class TestFlow(unittest.TestCase):

	def setUp(self):
		self.df = pd.DataFrame([10,100,60,40,200], columns=['frame.len'])
		self.f1 = Flow(self.df)

	def test_len(self):
		self.assertEqual(len(self.f1), 5)

	def test_mean(self):
		self.assertEqual(self.f1.mean_packet_size(), 82)

	def test_variance(self):
		self.assertEqual(self.f1.sizevar(), 5420)

if __name__ == '__main__':
	unittest.main()
