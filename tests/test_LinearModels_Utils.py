import unittest
import numpy as np

from CML_tool.LinearModels_Utils import GelmanScaler

class TestGelmanScaler(unittest.TestCase):
    def setUp(self):
        self.scaler = GelmanScaler()
        self.X = np.array([[1, 2, 3], 
                           [0, 2, 0], 
                           [1, 2, 1], 
                           [0, 2, 1]])
        self.X_reverted = self.X.copy()
        self.X_transformed = np.array([[1, 0, 0.802955], 
                                    [0, 0, -0.573539], 
                                    [1, 0, -0.114708], 
                                    [0, 0, -0.114708]])

    def test_fit(self):
        self.scaler.fit(self.X)
        np.testing.assert_array_almost_equal(self.scaler.mean_, np.array([0.5, 2, 1.25]), decimal=4)
        np.testing.assert_array_almost_equal(self.scaler.stdev_, np.array([0.5, 0, 1.089725]), decimal=4)
        np.testing.assert_array_almost_equal(self.scaler.constant_indices, np.array([1]), decimal=4)
        np.testing.assert_array_almost_equal(self.scaler.binary_indices, np.array([0]), decimal=4)

    def test_transform(self):
        self.scaler.fit(self.X)
        np.testing.assert_array_almost_equal(self.scaler.transform(self.X), self.X_transformed, decimal=4)

    def test_inverse_transform(self):
        self.scaler.fit(self.X)
        np.testing.assert_array_almost_equal(self.scaler.inverse_transform(self.X_transformed), self.X_reverted,decimal=4)
        
if __name__ == "__main__":
    unittest.main()