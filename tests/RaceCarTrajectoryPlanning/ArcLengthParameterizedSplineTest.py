import unittest
import src.Planning.RaceCarTrajectoryPlanning.ArcLengthParameterizedSpline as alps
import numpy as np

def arc_spline_position(spline, t):
    a = spline[0]
    b = spline[1]
    c = spline[2]
    d = spline[3]

    return a*t**3 + b*t**2 + c*t + d

class MyTestCase(unittest.TestCase):
    def assertSequenceAlmostEqual(self, seq1, seq2, decimal):
        self.assertEqual(len(seq1), len(seq2))
        for i in range(len(seq1)):
            self.assertAlmostEqual(seq1[i], seq2[i], decimal)

    def test_arclength_calculation(self):
        processor = alps.ArcLengthParameterizedSplines(dim=2)
        a = np.array([1, 0])
        b = np.array([2, 0])
        c = np.array([3, 1])
        d = np.array([4, 5])

        processor.add_spline([a, b, c, d])
        length = processor.get_spline_lengths()[0]

        self.assertAlmostEqual(length, 6.0929, 2)

    def test_get_spline_position_from_length(self):
        processor = alps.ArcLengthParameterizedSplines(dim=2, epsilon=0.001)
        a = np.array([1, 0])
        b = np.array([2, 0])
        c = np.array([3, 1])
        d = np.array([4, 5])

        position, tangent, t = processor._get_position_in_spline([a, b, c, d], 1)

        position_correct = np.array([4.96261, 5.267])
        tangent_correct = np.array([4.28187, 1])
        t_correct = 0.267
        self.assertSequenceAlmostEqual(position.tolist(), position_correct.tolist(), 2)
        self.assertSequenceAlmostEqual(tangent, tangent_correct, 2)
        self.assertAlmostEqual(t, t_correct, 2)

    def test_get_position_from_multiple_splines(self):
        processor = alps.ArcLengthParameterizedSplines(dim=2, epsilon=0.001)
        a1 = np.array([1, 0])
        b1 = np.array([2, 0])
        c1 = np.array([3, 1])
        d1 = np.array([4, 5])

        a2 = np.array([1, 0])
        b2 = np.array([2, 0])
        c2 = np.array([3, 1])
        d2 = np.array([14, 15])

        processor.add_spline([a1, b1, c1, d1])
        processor.add_spline([a2, b2, c2, d2])

        position, tangent, t, idx = processor._find_position_in_splines(7.0929)

        position_correct = np.array([14.96261, 15.267])
        tangent_correct = np.array([4.28187, 1])
        t_correct = 0.267
        idx_correct = 1

        self.assertSequenceAlmostEqual(position, position_correct, 2)
        self.assertSequenceAlmostEqual(tangent, tangent_correct, 2)
        self.assertAlmostEqual(t, t_correct, 2)
        self.assertAlmostEqual(idx, idx_correct, 2)

    def test_compute_arc_length_parameterized_spline(self):
        processor = alps.ArcLengthParameterizedSplines(dim=2, epsilon=0.001)
        a1 = np.array([1, 0])
        b1 = np.array([2, 0])
        c1 = np.array([3, 1])
        d1 = np.array([4, 5])

        a2 = np.array([3, 1])
        b2 = np.array([2, 0])
        c2 = np.array([10, 1])
        d2 = np.array([10, 6])

        processor.add_spline([a1, b1, c1, d1])
        processor.add_spline([a2, b2, c2, d2])

        lengths = processor.get_spline_lengths()
        spline_1_length = 6.093
        spline_2_length = 15.137

        self.assertAlmostEqual(lengths[0], spline_1_length, 2)
        self.assertAlmostEqual(lengths[1], spline_2_length, 2)

        # Total length 21.230, split into 4 segment,
        # each segment is of length 5.3075
        # corresponding parameters are 0, 0.9186, 0.399, 0.7428, 1
        # Corresponding spline indices are 0, 0, 1, 1, 1
        # Corresponding positions are (4, 5), (9.21859, 5.9186), (14.499, 6.46252), (19.761, 7.15264), (25, 8)
        num_segments = 4
        seg_length_correct = (spline_1_length + spline_2_length) / num_segments
        arc_splines, seg_length, ts, indices \
            = processor.compute_arc_length_parameterized_spline(num_segments)

        self.assertAlmostEqual(seg_length, 5.3075, 2)

        self.assertAlmostEqual(ts[0], 0, 2)
        self.assertAlmostEqual(ts[1], 0.9186, 2)
        self.assertAlmostEqual(ts[2], 0.399, 2)
        self.assertAlmostEqual(ts[3], 0.7428, 2)
        self.assertAlmostEqual(ts[4], 1, 2)

        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[1], 0)
        self.assertEqual(indices[2], 1)
        self.assertEqual(indices[3], 1)
        self.assertEqual(indices[4], 1)

        self.assertSequenceAlmostEqual(arc_spline_position(arc_splines[0], 0), [4, 5], 2)
        self.assertSequenceAlmostEqual(arc_spline_position(arc_splines[0], seg_length_correct), [9.21859, 5.9186], 2)
        self.assertSequenceAlmostEqual(arc_spline_position(arc_splines[1], 0), [9.21859, 5.9186], 2)
        self.assertSequenceAlmostEqual(arc_spline_position(arc_splines[1], seg_length_correct), [14.499, 6.46252], 2)
        self.assertSequenceAlmostEqual(arc_spline_position(arc_splines[2], 0), [14.499, 6.46252], 2)
        self.assertSequenceAlmostEqual(arc_spline_position(arc_splines[2], seg_length_correct), [19.761, 7.15264], 2)
        self.assertSequenceAlmostEqual(arc_spline_position(arc_splines[3], 0), [19.761, 7.15264], 2)
        self.assertSequenceAlmostEqual(arc_spline_position(arc_splines[3], seg_length_correct), [25, 8], 2)

if __name__ == '__main__':
    unittest.main()
