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



class Test_chol(TestCase):
    def test_chol(self):
        from non_neg_ls.solve_chol import add_rc
        import numpy as np
        X = np.random.rand(20, 4)
        XTX = X.T @ X
        Ltrue = np.linalg.cholesky(XTX)
        L = np.zeros_like(Ltrue)
        perm = np.arange(4, dtype='int32')
        n = 0
        for i in range(4):
            add_rc(L, perm, n, i, XTX[i,:])
            n += 1
        np.testing.assert_allclose(L, Ltrue)
