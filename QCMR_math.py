# -*- coding: utf-8 -*-
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
from __future__ import print_function

"""
"""
import numpy as np
from scipy.optimize import curve_fit
import scipy.ndimage as scind
import scipy.stats

class RigidTransform():
    def getRMSE(self,inA,inB):
        if self.rotmatrix is None or self.translation is None:
            return

        assert len(inA) == len(inB)
        A = inA
        if not isinstance(inA,np.matrix):
            A = np.mat(inA)

        if not self.scaling is None:
            newA = self.apply_scaling(A)
            A = np.mat(newA)

        B = inB
        if not isinstance(inB,np.matrix):
            B = np.mat(inB)

        N = A.shape[0] # total points

        A2 = (self.rotmatrix*A.T) + np.tile(self.translation, (1, N))
        A2 = A2.T
        
        # Find the error
        err = A2 - B
        
        err = np.multiply(err, err)
        err = np.sum(err)
        rmse = np.sqrt(err/N)
        return rmse
    
    def apply_scaling(self,inA):
        A = inA
        if not isinstance(inA,np.matrix):
            A = np.mat(inA)

        if not self.scaling is None:
            newA = []
            for a in A:
                newA.append( [c+self.scaling*(x-c) for x,c in zip(a.tolist()[0],self.scalingcenter)] )
            A = np.mat(newA)
        return A.tolist()
        
    def apply(self,inA):
        if self.rotmatrix is None or self.translation is None:
            return

        A = inA
        if not isinstance(inA,np.matrix):
            A = np.mat(inA)
            
        if self.allowscaling and not self.scaling is None:
            newA = self.apply_scaling(A)
            A = np.mat(newA)
            

        N = A.shape[0] # total points
        A2 = (self.rotmatrix*A.T) + np.tile(self.translation, (1, N))
        A2 = A2.T
        if len(np.shape(inA)) == 1:
            return A2.tolist()[0]
        else:
            return A2.tolist()
        
    def getRotationDeg(self):
        if self.rotmatrix is None:
            return

        return 180/np.pi*np.arctan2(self.rotmatrix[0,1],self.rotmatrix[0,0])

    def getRotationRad(self):
        if self.rotmatrix is None:
            return

        return np.arctan2(self.rotmatrix[0,1],self.rotmatrix[0,0])
        
    def getRotation(self):
        return self.getRotationRad()
    def getShift(self):
        return self.translation.T.tolist()[0]
    
    def getTranslation(self):
        if self.translation is None:
            return
        return self.translation
    
    def __init__(self,fromA,toB,allowscaling=False):
        self.rotmatrix = None
        self.translation = None
        self.scaling = None
        self.scalingcenter = None
        
        self.allowscaling = allowscaling
        
        if not fromA is None and not toB is None:
            self.rigid_transform_3D(fromA,toB)
        
    def calc_scaling(self,inA,inB):
        # Input: expects Nx3 matrix of points
        # returns a scaling factor from A to B
        assert len(inA) == len(inB)
        A = inA
        if not isinstance(inA,np.matrix):
            A = np.mat(inA)
        B = inB
        if not isinstance(inB,np.matrix):
            B = np.mat(inB)

        N = A.shape[0] # total points
    
        centroid_A = np.mean(A, axis=0).tolist()[0]
        centroid_B = np.mean(B, axis=0).tolist()[0]

        arad = -1.
        for a in A:
            arad = max(arad,np.sqrt(np.sum( [(x1-x0)**2. for x1,x0 in zip(a.tolist()[0],centroid_A)] )))

        brad = -1.
        for b in B:
            brad = max(brad,np.sqrt(np.sum( [(x1-x0)**2. for x1,x0 in zip(b.tolist()[0],centroid_B)] )))
            
        return brad/arad,centroid_A
    
    def rigid_transform_3D(self,inA, inB):
        # Input: expects Nx3 matrix of points
        # Returns R,t
        # R = 3x3 rotation matrix
        # t = 3x1 column vector

        assert len(inA) == len(inB)
        A = inA
        if not isinstance(inA,np.matrix):
            A = np.mat(inA)
        B = inB
        if not isinstance(inB,np.matrix):
            B = np.mat(inB)

        if self.allowscaling:
            self.scaling,self.scalingcenter = self.calc_scaling(A,B)
            A = self.apply_scaling(A)
            if not isinstance(A,np.matrix):
                A = np.mat(A)
        else:
            self.scaling = None
            
        N = A.shape[0] # total points
    
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        
        # centre the points
        AA = A - np.tile(centroid_A, (N, 1))
        BB = B - np.tile(centroid_B, (N, 1))
    
        # dot is matrix multiplication for array
        H = np.transpose(AA) * BB
    
        U, S, Vt = np.linalg.svd(H)
    
        R = Vt.T * U.T
    
        # special reflection case
        if np.linalg.det(R) < 0:
            print("Reflection detected")
            Vt[2,:] *= -1
            R = Vt.T * U.T
    
        t = -R*centroid_A.T + centroid_B.T
    
        self.translation = t
        self.rotmatrix = R

    def test(self):
        # Test with random data
        
        # Random rotation and translation
        R = np.mat(np.random.rand(3,3))
        t = np.mat(np.random.rand(3,1))
        #t = np.mat([[0.],[0.],[0.]])
    
        # make R a proper rotation matrix, force orthonormal
        U, S, Vt = np.linalg.svd(R)
        R = U*Vt
        
        # remove reflection
        if np.linalg.det(R) < 0:
            Vt[2,:] *= -1
            R = U*Vt
        
        # number of points
        n = 10
        
        A = np.mat(np.random.rand(n,3));
        B = R*A.T + np.tile(t, (1, n))
        B = B.T
        
        # recover the transformation
        #ret_R, ret_t = 
        self.rigid_transform_3D(A, B)
        ret_R = self.rotmatrix
        ret_t = self.translation
        
        A2 = (ret_R*A.T) + np.tile(ret_t, (1, n))
        A2 = A2.T
        
        # Find the error
        err = A2 - B
        
        err = np.multiply(err, err)
        err = np.sum(err)
        rmse = np.sqrt(err/n)
        
        print("Points A")
        print(A)
        print("")
        
        print("Points B")
        print(B)
        print("")
        
        print("Rotation")
        print('  true')
        print(R)
        print(' found')
        print(ret_R)
        print("")
        
        print("Translation")
        print('  true')
        print(t)
        print(' found')
        print(ret_t)
        print("")
        
        print("RMSE:", rmse)
        print("If RMSE is near zero, the function is correct!")
        print('SCALING:',self.calc_scaling(A,B))

    def test2(self):
        A = [[40.96, 127.5], [127.5, 215.04], [215.04, 127.5]]
        B = [[43.210000000000001, 127.25], [129.0, 213.03999999999999], [214.53999999999999, 127.0]]

        # recover the transformation
        self.rigid_transform_3D(A, B)
        
        print("Points A")
        print(A)
        print("")

        print("Points B")
        print(B)
        print("")

        print("Rotation")
        print(self.rotmatrix)
        print("")
        print("angle:", self.getRotationDeg())
        print("Translation")
        print(self.translation)
        print("")
        print('RMSE',self.getRMSE(A, B))
        print('SCALING:',self.calc_scaling(A,B))
        print(self.apply(A))

def FindCenters2D(pts,datain,distpx,discpx,minimod=False):
    error = True
    searchrad = int(distpx+.5)
    searchrad = max(1,searchrad)

    # scale by 4x4 by copies for 1/4 pix accuracy
    multip = 4
    data = np.kron(datain,np.ones((multip,multip)))

    # smoothing to get rid of noise and give max respons over avg disc
    sigma = multip*discpx/2.
    data = scind.gaussian_filter(data.astype(float), sigma,mode='constant')
    searchrad *= multip
    """"
    # scale by 2x2 by copies
    a = np.array([[1, 1],
                  [0, 1]])
    n = 2
    np.kron(a, np.ones((n,n)))
    """
    
    widthpx = np.shape(data)[0] ## width/height in pixels
    heightpx = np.shape(data)[1]
    import matplotlib.pyplot as plt

    for y in range(len(pts)):
        for x in range(len(pts[y])):
            rp = pts[y][x]
            if(len(rp)==0):
                continue
            x0 = multip*rp[0]
            minx = max(0,x0-searchrad)
            maxx = min(widthpx-2,x0+searchrad)
            y0 = multip*rp[1]
            miny = max(0,y0-searchrad)
            maxy = min(heightpx-2,y0+searchrad)
            cropped = data[int(minx):int(maxx+1),int(miny):int(maxy+1)]
            #plt.figure()
            #plt.title(rp)
            #plt.imshow(cropped)
            #plt.show()
            if(minimod == True):
                (x1,y1) = np.unravel_index(cropped.argmin(),cropped.shape)
            else:
                (x1,y1) = np.unravel_index(cropped.argmax(),cropped.shape)
            x1 += minx
            y1 += miny
            rp[0] = x1/multip
            rp[1] = y1/multip  

    error = False
    return error,pts


# Define model function to be used to fit to the data 
def gauss(x, *p):
    A, mu, sigma,c = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))+c

def GaussianFit(data):
    error = True
    
    # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
    p0 = [np.max(data)-np.min(data), len(data)/2., 1.,data[0]]

    pos = range(len(data))
    coeff, var_matrix = curve_fit(gauss, pos, data, p0=p0)
    error = False
    return error, coeff

def LinearFit(ydata):
    error = True
    pos = range(len(ydata))
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(pos,ydata)

    return error,(slope, intercept, r_value, p_value, std_err)

def AreaUnderCurve(freqs,merits,maxfreq=-1,normalized=True):
    area = 0.
    if(len(freqs) == 0 or len(merits) == 0):
        return area
    for k in range(0,len(freqs)-1):
        if maxfreq >0. and freqs[k]<= maxfreq:
            y1 = merits[k]
            y2 = merits[k+1]
            dx = freqs[k+1]-freqs[k]
            if(y2<y1):
                y2 = merits[k]
                y1 = merits[k+1]
            area += dx*(y1+(y2-y1)/2.)

        if maxfreq >0. and freqs[k]< maxfreq and freqs[k+1]>maxfreq:
            dx = maxfreq-freqs[k]
            y1 = merits[k]
            y2 = y1+(maxfreq-freqs[k])/(freqs[k+1]-freqs[k])*(merits[k+1]-merits[k])

            if(y2<y1):
                yswap = y2
                y2 = merits[k]
                y1 = yswap
            area += dx*(y1+(y2-y1)/2.)

    if normalized:
        if maxfreq < 0.:
            area /= freqs[-1]
        else:
            area /= maxfreq

    return area
