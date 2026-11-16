"""Public HW07 examples. Run: python3 -m unittest -v test_hw07."""
import unittest
import numpy as np
import LogisticRegression as lr
from run_LR import confusion_matrix, parse_csv
from pathlib import Path
import tempfile

class PublicTests(unittest.TestCase):
    def test_bias(self):
        X = np.array([[2., 3.], [4., 5.]])
        np.testing.assert_array_equal(lr.add_ones(X), [[1,2,3],[1,4,5]])
        np.testing.assert_array_equal(X, [[2,3],[4,5]])

    def test_sigmoid(self):
        self.assertAlmostEqual(lr.logistic(np.array([1.,2.]), np.array([0.,0.])), .5)
        self.assertAlmostEqual(lr.logistic(np.array([1.]), np.array([np.log(3)])), .75)

    def test_extreme_sigmoid(self):
        for z in (-1000.,1000.):
            p=lr.logistic(np.array([1.]), np.array([z]))
            self.assertTrue(np.isfinite(p) and 0 <= p <= 1)

    def test_loss(self):
        print(-2*np.log(.75))
        self.assertAlmostEqual(lr.costJ(np.array([0,1]),np.array([.25,.75])), -2*np.log(.75))

    def test_loss_clipping(self):
        self.assertTrue(np.isfinite(lr.costJ(np.array([0,1]),np.array([1.,0.]))))

    def test_prediction(self):
        np.testing.assert_allclose(lr.prediction(np.array([[0.],[1.]]),np.array([0.,np.log(3)])),[.5,.75])

    def test_threshold(self):
        np.testing.assert_array_equal(lr.threshold(np.array([.2,.5,.9])),[0,1,1])

    def test_accuracy(self):
        self.assertAlmostEqual(lr.accuracy(np.array([[-1.],[1.]]),np.array([0.,1.]),np.array([0,1])),1.)

    def test_confusion(self):
        np.testing.assert_array_equal(confusion_matrix(np.array([0,0,1,1,1]),np.array([0,1,0,1,1])),[[1,1],[1,2]])

    def test_single_update(self):
        np.testing.assert_allclose(lr.fit_SGD(np.array([[2.]]),np.array([1]),.2,tmax=1),[.1,.2])

    def test_invalid_parameters(self):
        for kwargs in ({'alpha':0},{'alpha':-.1},{'alpha':.1,'eps':-1},{'alpha':.1,'tmax':0}):
            with self.assertRaises(ValueError): lr.fit_SGD(np.array([[1.]]),np.array([1]),**kwargs)

    def test_learning_and_no_mutation(self):
        X=np.array([[-2.],[-1.],[1.],[2.]])
        y=np.array([0,0,1,1]); original=X.copy()
        coef=lr.fit_SGD(X,y,.1,tmax=100,eps=0,seed=7)
        self.assertEqual(lr.accuracy(X,coef,y),1.)
        np.testing.assert_array_equal(X,original)
        np.testing.assert_array_equal(y,[0,0,1,1])

    def test_csv(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.csv'; p.write_text('2.5,0\n3.5,1\n')
            X,y=parse_csv(p)
            np.testing.assert_array_equal(X,[[2.5],[3.5]])
            np.testing.assert_array_equal(y,[0,1])

if __name__ == '__main__': unittest.main()
