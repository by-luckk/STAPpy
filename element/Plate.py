import sys
sys.path.append('../')
import numpy as np
from element.Element import CElement

class CPlate(CElement):
    """ Plate Element class """
    def gauss(self, ngp):
        """
        Get Gauss points in the parent element domain [-1, 1] and
        the corresponding weights.
        Args:
            ngp : (int) number of Gauss points.
        Returns: w,gp
            w  : weights.
            gp : Gauss points in the parent element domain.
        """
        gp = None
        w = None
        if ngp == 1:
            gp = 0
            w = 2
        elif ngp == 2:
            gp = [-0.57735027, 0.57735027]
            w = [1, 1]
        elif ngp == 3:
            gp = [-0.7745966692, 0.7745966692, 0.0]
            w = [0.5555555556, 0.5555555556, 0.8888888889]
        else:
            raise ValueError("The given number (ngp = {}) of Gauss points is too large and not implemented".format(ngp))
        return w, gp
    
    