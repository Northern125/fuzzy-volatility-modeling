import unittest

from numpy import array

from src.fuzzy_inference_systems.takagi_sugeno import TSFuzzyInferenceSystem


class TSFISTestCase(unittest.TestCase):
    def test_2_clusters_same_consequent_params(self):
        clusters_params = {
            'centers': [array([1, 2]), array([3, 4])],
            'cov_matrices': [array([[1, 0],
                                   [0, 1]]),
                             array([[1, 0],
                                    [0, 1]])
                             ]
        }
        data_to_cluster = array([1, 2])

        fis = TSFuzzyInferenceSystem(membership_function='gaussian',
                                     clusters_params=clusters_params,
                                     data_to_cluster=data_to_cluster,
                                     normalize=True)
        fis.input = array([1, 2, 3])
        fis.consequent_params = array([[10, 20, 30, 40],
                                       [10, 20, 30, 40]])

        fis.calc_membership_degrees()
        fis.calc_consequents(add_to_hist=True)
        fis.defuzzify(add_to_hist=True)

        self.assertEqual(fis.output, 210)


if __name__ == '__main__':
    unittest.main()
