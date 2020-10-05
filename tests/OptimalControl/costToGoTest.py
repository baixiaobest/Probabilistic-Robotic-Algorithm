import unittest
import numpy as np
import src.OptimalControl.CostToGo as ctg

class MyTestCase(unittest.TestCase):
    def test_empty_cost_to_go(self):
        configs = [{"min": -20, "max": 50, "resolution": 0.1},
                   {"min": 10, "max": 50, "resolution": 0.1},
                   {"min": 0 , "max": np.pi, "resolution": 0.01}]
        cost_to_go = ctg.CostToGo(configs)

        state_1 = np.array([-20, 10, 0])
        state_2 = np.array([50, 50, np.pi])
        state_3 = np.array([4.76, 14.83, 1.763])

        indices_1 = [0, 0, 0]
        indices_2 = [50, 50, 50]

        self.assertEqual(0, cost_to_go.get_cost(state_1))
        self.assertEqual(0, cost_to_go.get_cost(state_2))
        self.assertEqual(0, cost_to_go.get_cost(state_3))
        self.assertEqual(0, cost_to_go.get_cost_by_indices(indices_1))
        self.assertEqual(0, cost_to_go.get_cost_by_indices(indices_2))

    def test_setting_cost(self):
        configs = [{"min": -20, "max": 50, "resolution": 0.1},
                   {"min": 10, "max": 50, "resolution": 0.1},
                   {"min": 0, "max": np.pi, "resolution": 0.01}]
        cost_to_go = ctg.CostToGo(configs)

        state_1 = np.array([-20, 10, 0])
        cost_1 = 10
        state_2 = np.array([50, 50, np.pi])
        cost_2 = 94.543
        state_3 = np.array([4.76, 14.83, 1.763])
        cost_3 = 82.34

        indices_1 = [0, 0, 0]
        cost_4 = 934.7
        indices_2 = [50, 50, 50]
        cost_5 = 234.234

        cost_to_go.set_cost(state_1, cost_1)
        cost_to_go.set_cost(state_2, cost_2)
        cost_to_go.set_cost(state_3, cost_3)
        cost_to_go.set_cost_by_indices(indices_1, cost_4)
        cost_to_go.set_cost_by_indices(indices_2, cost_5)

        self.assertEqual(cost_4, cost_to_go.get_cost(state_1))
        self.assertEqual(cost_2, cost_to_go.get_cost(state_2))
        self.assertEqual(cost_3, cost_to_go.get_cost(state_3))
        self.assertEqual(cost_4, cost_to_go.get_cost_by_indices(indices_1))
        self.assertEqual(cost_5, cost_to_go.get_cost_by_indices(indices_2))

    def test_indexing(self):
        configs = [{"min": -20, "max": 50, "resolution": 0.1},
                   {"min": 10, "max": 50, "resolution": 0.1},
                   {"min": 0, "max": np.pi, "resolution": 0.01}]
        cost_to_go = ctg.CostToGo(configs)

        indices = [0, 0, 0]
        state = np.array([-20, 10, 0])

        indices2 = [1, 1, 1]
        state2 = np.array([-19.9, 10.1, 0.01])

        self.assertTrue(np.allclose(state, cost_to_go.get_state_from_indices(indices)))
        self.assertTrue(np.allclose(state2, cost_to_go.get_state_from_indices(indices2)))


if __name__ == '__main__':
    unittest.main()
