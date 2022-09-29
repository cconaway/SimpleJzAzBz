
import numpy as np
import math

class ColorConverter(object):

    #CONSTANTS & CONVERSION MATRICES

    #For JzAzBz Calculations
    b=1.15
    g=0.66
    d = -0.56
    n = 2610 / (2**14)
    p = (1.7) * 2523/ (2**5) 
    c1 = 3424/ (2**12)
    c2 = 2413/ (2**7) 
    c3 = 2392/ (2**7)

    #Array for XYZ to LMS conversion
    M1 = np.array([[0.41478972, 0.579999, 0.014648],
                  [-0.20151, 1.120649, 0.0531008],
                  [-0.0166008, 0.2648, 0.6684799]])

    #Array for L'M'S' to IzAzBz 
    M2 = np.array([[0.5, 0.5, 0],
                [3.524000,  -4.066708, 0.542708],
                [0.199076, 1.096799,  -1.295875]])

    #Array for XYZ_D65 to sRGB  
    M3 = np.array([[3.2404542, -1.5371385, -0.4985314],
                [-0.9692660,  1.8760108,  0.0415560],
                [0.0556434, -0.2040259,  1.0572252]])

    #Conversion from sRGB to XYZ65
    M4 = np.array([[0.4124564, 0.3575761, 0.1804375],
                 [0.2126729, 0.7151522, 0.0721750],
                 [0.0193339, 0.1191920, 0.9503041]])

    """Hidden Methods for internal Calculation!"""
    #Perceptual Quantizer -> function for curve which represents dynamic transform of cone responses.
    def _PQ(self, X, lum=10000):
        #Converts LMS into L'M'S'

        xn = np.sign(X / lum) * (np.abs(X / lum) ** self.n) 
        a = self.c1 + (self.c2 * xn)
        b = 1 + (self.c3 * xn)
    
        LMS_prime = np.sign(a/b) * (np.abs(a/b) ** self.p)
        
        return LMS_prime

    #Inverse PQ for JzAzBz to Xyz
    def _invPQ(self, X, lum=10000):
        #Converts L'M'S' into LMS
        xn = np.sign(X) * (np.abs(X)**(1/self.p)) #x.^(1/p)
        a = self.c1-xn
        b = self.c3*xn-self.c2
    
        LMS = lum * ( np.sign(a/b) * (np.abs(a/b)**(1/self.n)))
    
        return LMS

    """Instance Methods for Conversions!"""
    #XYZ_D65 to JzAzBz converter
    def xyz65_2_jzazbz(self, XYZ_D65, lum=10000):
        """
        Input is an n x 3 np.matrix!!!

        ex = np.array([[0.0, 0.0, 0.0]])
        
        """
        
        XYZ_D65[:,1] = self.g * XYZ_D65[:,1] - np.multiply((self.g-1),XYZ_D65[:,0])
        XYZ_D65[:,0] = self.b * XYZ_D65[:,0] - np.multiply((self.b-1),XYZ_D65[:,2])

        LMS_prime = self._PQ(XYZ_D65 @ self.M1.T, lum=lum)
        IzAzBz = LMS_prime @ self.M2.T  
        
        Iz = IzAzBz[:,0]
        Jz = ((1+self.d) * Iz) / (1+self.d*Iz)-1.6295499532821566e-11
        JzAzBz =  np.array([Jz, IzAzBz[:,1], IzAzBz[:,2]]).T 
        
        return JzAzBz
    
    #JzAzBz to XYZ_D65 converter
    def jzazbz_2_xyz65(self, JzAzBz, lum=10000):
        """
        Input is an n x 3 np.matrix!!!

        ex = np.array([[0.0, 0.0, 0.0]])
        
        """
        
        JzAzBz[:,0] = JzAzBz[:,0]+1.6295499532821566e-11
        Iz = JzAzBz[:,0] / (1 + self.d - (self.d*JzAzBz[:,0]))
        
        IzAzBz = np.array([Iz, JzAzBz[:,1], JzAzBz[:,2]]).T

        XYZ_D65 = self._invPQ(IzAzBz @ np.linalg.inv(self.M2).T, lum=lum) @ np.linalg.inv(self.M1).T
        
        XYZ_D65[:,0] = (XYZ_D65[:,0] + np.multiply((self.b-1),XYZ_D65[:,2])) / self.b 
        XYZ_D65[:,1] = (XYZ_D65[:,1] + np.multiply((self.g-1),XYZ_D65[:,0])) / self.g

        return XYZ_D65

    #JzAzBz to JzCzHz converter
    def jzazbz_2_jzczhz(self, JzAzBz):
        """
        Input is an n x 3 np.matrix!!!

        ex = np.array([[0.0, 0.0, 0.0]])
        """
        
        #math is used in place of numpy for extremely large values, and type long support

        Cz = np.array([math.sqrt( (JzAzBz[:,1]**2) + (JzAzBz[:,2]**2) )])
        Hz = np.arctan2(JzAzBz[:,2], JzAzBz[:,1])
        
        return np.array([JzAzBz[:,0], Cz, Hz]).T
    
    
    #JzCzHz to JzAzBz converter
    def jzczhz_2_jzazbz(self, JzCzHz):
        
        return np.array([JzCzHz[:,0], JzCzHz[:,1]*np.cos(JzCzHz[:,2]), JzCzHz[:,1]*np.sin(JzCzHz[:,2])]).T    

    #Calculate the Delta E between two JzAzBz arrays!
    def jzazbz_deltaE(self, JzAzBz1, JzAzBz2):
        """
        Input is an n x 3 np.matrix!!!

        ex = np.array([[0.0, 0.0, 0.0]])
        
        """

        JzCzHz1 = self.jzazbz_2_jzczhz(JzAzBz1)
        JzCzHz2 = self.jzazbz_2_jzczhz(JzAzBz2)
    
        d_j = JzCzHz2[:,0] - JzCzHz1[:,0]
        d_c = JzCzHz2[:,1] - JzCzHz1[:,1]
        d_h = 2 * math.sqrt(JzCzHz2[:,1] * JzCzHz1[:,1]) * np.sin( (JzCzHz2[:,2] - JzCzHz1[:,2]) / 2 ) 
        
        return np.sqrt( (d_j**2) + (d_c**2) + (d_h**2))

    #Convert XYZ_D65 to sRGB
    def xyz65_2_srgb(self, XYZ_D65):
        """
        Input is an n x 3 np.matrix!!!

        ex = np.array([[0.0, 0.0, 0.0]])
        
        """
        rgb = XYZ_D65 @ self.M3.T
        
        #need to make adjustments to nonlinearity of srgb,
        # will instead use linear rgb for now
        #rgb[:,0] = self._nonlinear_xyz_rgb(rgb[:,0])
        #rgb[:,1] = self._nonlinear_xyz_rgb(rgb[:,1])
        #rgb[:,2] = self._nonlinear_xyz_rgb(rgb[:,2])

        #Note that the method of rounding will impact interpolated RGBs
        return np.round(rgb * 255)
    
    #Convert sRGB to XYZ_D65
    def srgb_2_xyz65(self, sRGB):
        """
        Input is an n x 3 np.matrix!!!

        ex = np.array([[0.0, 0.0, 0.0]])
        
        """
        sRGB = sRGB/255
        
        #sRGB[:,0] = self._nonlinear_rgb_xyz(sRGB[:,0])
        #sRGB[:,1] = self._nonlinear_rgb_xyz(sRGB[:,1])
        #sRGB[:,2] = self._nonlinear_rgb_xyz(sRGB[:,2])
        
        
        return (self.M4 @ sRGB.T).T
    
    def _nonlinear_xyz_rgb(self, x):
        for i, j in enumerate(x):
            if j <= 0.0031308:
                x[i] = j * 12.92
            else:
                x[i] = 1.055 * (j ** (1/2.4)) - 0.055
        
        return x
    
    def _nonlinear_rgb_xyz(self, x):
        if x <= 0.04045:
            return x / 12.92
        else:
            return (np.sign((x+0.055)/1.055) * np.power(np.abs(x+0.055)/1.055, 2.4))

