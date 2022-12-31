from unittest import TestCase


class Test_L_calc(TestCase):
    def test_solve_lower(self):
        import numpy as np
        from non_neg_ls import solve_lower
        L = np.array([[2.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 2.0]])
        perm = np.arange(3, dtype='int32')
        n = 3
        y = np.ones(3)
        x = np.empty((3))
        solve_lower(L, perm, n, y, x)
        np.testing.assert_allclose(x, [0.5, 0.25, -0.25])

    def test_solve_upper(self):
        import numpy as np
        from non_neg_ls import solve_upper
        L = np.array([[2.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 2.0]])
        perm = np.arange(3, dtype='int32')
        n = 3
        y = np.array([1.0, 2.0, 4.0])
        x = np.empty((3), dtype=float)
        solve_upper(L, perm, n, y, x)
        np.testing.assert_allclose(x, [-1.0, -1.0, 2.0])
    
    def test_add_rc(self):
        import numpy as np
        from non_neg_ls import add_rc
        A = np.array([[1,2],[3,4], [5,6]], float)
        ATA = A.T @ A
        L = np.empty((2, 2))
        perm = np.array([1, 0], 'int32')
        n = np.array(0, 'int32')
        p = np.array(1, 'int32')
        Acol = ATA[1]
        add_rc(L, perm, n, p, Acol)
        assert L[1, 1] == np.sqrt(56)


class Test_chol(TestCase):
    def test_chol(self):
        from non_neg_ls import add_rc
        import numpy as np
        X = np.random.rand(20, 4)
        XTX = X.T @ X
        Ltrue = np.linalg.cholesky(XTX)
        L = np.zeros_like(Ltrue)
        perm = np.arange(4, dtype='int32')
        n = 0
        for i in range(4):
            add_rc(L, perm, n, i, XTX[i, :])
            n += 1
        np.testing.assert_allclose(L, Ltrue,  rtol=10**-5)

    def test_chol_mult(self):
        from non_neg_ls import add_rc, mult_LLT
        import numpy as np
        X = np.random.rand(20, 4)
        XTX = X.T @ X
        Ltrue = np.linalg.cholesky(XTX)
        L = np.zeros_like(Ltrue)
        perm = np.arange(4, dtype='int32')
        n = 0
        for i in range(4):
            add_rc(L, perm, n, i, XTX[i, :])
            n += 1
        r = mult_LLT(L, perm, n)
        np.testing.assert_allclose(r, XTX)


class Test_nn_ls(TestCase):
    def test_small_rand_runs(self):
        import numpy as np
        import non_neg_ls
        A = np.random.rand(20, 4)
        y = np.ones(20, float)
        x = non_neg_ls.nn_least_squares(A, y)
        assert len(x) == 4
