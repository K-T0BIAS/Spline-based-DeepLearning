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

class CTensor_Test(unittest.TestCase):
    
    def CTensor_init_Test(self):
        a = PySplineNetLib.CTensor([[1,2,3],[4,5,6]])
        self.assertAlmostEqual([1,2,3,4,5,6], a.data())
        self.assertAlmostEqual([2,3], a.data())
        b = PySplineNetLib.CTensor([6,5,4,3,2,1],[3,2])
        self.assertAlmostEqual([6,5,4,3,2,1], b.data())
        self.assertAlmostEqual([3,2], b.shape())
        c = PySplineNetLib.CTensor(a)
        self.assertAlmostEqual([1,2,3,4,5,6], c.data())
        self.assertAlmostEqual([2,3], c.shape())
        
    def CTensor_math_Test(self):
        a = PySplineNetLib.CTensor([[1,2,3],[4,5,6]])
        b = PySplineNetLib.CTensor([[6,5,4],[3,2,1]])
        
        c = a + b;
        self.assertAlmostEqual([7,7,7,7,7,7], c.data())
        self.assertAlmostEqual([2,3], c.shape())
        
        b.transpose()
        d = a * b;
        self.assertAlmostEqual([28.0, 10.0, 73.0, 28.0], d.data())
        self.assertAlmostEqual([2,2], d.shape())
        
        b.transpose()
        e = a - b;
        self.assertAlmostEqual([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0], e.data())
        self.assertAlmostEqual([2,2], e.shape())
        
    def Ctensor_grad_Test(self):
        a = PySplineNetLib.CTensor([[2,2,2],[2,2,2]])
        b = PySplineNetLib.CTensor([[1,2],[3,4],[5,6]])
        c = PySplineNetLib.CTensor([[0.5,0.5],[0.5,0.5]])
        d = a * b + c