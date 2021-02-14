import unittest
import sys

sys.path.append("../src/")
sys.path.append("src/")

from src.analysis_utils import *


class Test_analysis_utils(unittest.TestCase):

	def test_rmse(self):
		""" Tests RMSE calc """
		targets=[1,2,3,4,5]
		preds = [1.6,2.5,2.9,3,4.1]
		self.assertEqual(rmse(preds, targets), 0.6971370023173351, f"Should be {0.6971370023173351}")

	def test_mse(self):
		""" Tests MSE calc """
		targets=[11,20,19,17,10]
		preds = [12,18,19.5,18,9]
		self.assertEqual(rmse(preds, targets), 1.2041594578792296, f"Should be {1.2041594578792296}")

	def test_error_grad_drop(self):
		""" Tests gradient error drop"""
		data = [10,9,8,7,6,5,4,3,2,1]
		self.assertEqual(compute_grad(data), 10, f"Should be {10}")




if __name__ == '__main__':
    unittest.main()