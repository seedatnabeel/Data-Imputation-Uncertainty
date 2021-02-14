import unittest
import sys

sys.path.append("../src/")
sys.path.append("src/")

from src.utils import *
import numpy as np

class Test_utils(unittest.TestCase):

	def test_get_missing_indices(self):
		""" Tests getting missing data indices """

		# 1-D test
		one_d=np.array([1,0,1])
		self.assertEqual(get_missing_indices(one_d), 1)

		# 2-D test
		two_d=np.array([[0,1,1], [1,1,1], [1,1,1]])
		missing_2d = get_missing_indices(two_d)
		assert missing_2d[0][0]==0 and missing_2d[0][1]==0

	def test_norm_data(self):
		""" Tests normalizing the data """
		data=[1,2,3,4,5]

		scaled = normdata(np.array(data).reshape(-1,1))

		self.assertEqual(np.max(scaled),1, "Max should be 1")
		self.assertEqual(np.min(scaled),0, "Min should be 1")

	def test_prior(self):
		""" Tests the application of the prior """
		arr = np.array([1,2,3])
		mask = np.array([1,0,1])
		prior_type='Zero'
		out = prior(arr, mask, prior_type)
		


if __name__ == '__main__':
    unittest.main()