from unittest import TestCase


class Test_L_calc(TestCase):
    def test_solve_lower(self):
        import numpy as np
        from non_neg_ls.solve_chol import solve_lower
        L = np.array([[2.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 2.0]])
        perm = np.arange(3, dtype='int32')
        n = 3
        y = np.ones(3)
        x = np.empty((3))
        solve_lower(L, perm, n, y, x)
        np.testing.assert_allclose(x, [0.5, 0.25, -0.25])

    def test_solve_upper(self):
        import numpy as np
        from non_neg_ls.solve_chol import solve_upper
        L = np.array([[2.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 2.0]])
        perm = np.arange(3, dtype='int32')
        n = 3
        y = np.array([1.0, 2.0, 4.0])
        x = np.empty((3), dtype=float)
        solve_upper(L, perm, n, y, x)
        np.testing.assert_allclose(x, [-1.0, -1.0, 2.0])
