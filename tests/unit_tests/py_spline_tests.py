import PySplineNetLib
import unittest

class Spline_Test(unittest.TestCase):
    
    def Spline_init_Test(self):
        A = PySplineNetLib.spline([[0,0],[0.5,1],[1,2]],[[0,0,0,0],[0,0,0,0]])
        A.interpolation()
        a : float = A.forward(0.25)
        self.assertAlmostEqual(0.5, a, delta = 0.000001)
        a_y : float = A.backward(0.25, 0, 1)
        #returns A.forward(0.25)=0.5 - y = 0 + d_y = 0 -> 0.5 - 1 = -0.5
        self.assertAlmostEqual(-0.5, a_y, delta = 0.000001)
        A.apply_grad(1) #applies the gradient with factor 1.0 (moves y_i at x_i > 0.25 by -1 *grad {same as sign(grad)})
        A.interpolation() #fimds new params for the new spline
        self.assertAlmostEqual([0.0, 3.5, 0.0, -2.0],[1.5, 2.0, -3.0, 2.0], A.get_params())
        self.assertAlmostEqual([[0.0, 0.0], [0.5, 1.5], [1.0, 2.0]], A.get_points())
        