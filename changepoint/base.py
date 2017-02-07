from GPy.kern import Kern
from GPy.kern.src.kern import CombinationKernel
from GPy.core.parameterization import Param
from GPy.core.parameterization.transformations import Logexp
import numpy as np

CP_TOL = 1e-5

class Changepoint(CombinationKernel):
    """Kernel for a changepoint at position xc """

    def __init__(self,k1,k2=None,kc=1.,xc=np.array([[0]]),cpDim=0, changepointParameter=False):
        """
        arguments:
            k1, k2: GPy.kern.Kernel
            kc: float, covariance at the changepoint
            xc: np.array, position of changepoint(s)
            cpDim: int, dimension that changepoint exists on
            changepointParameter: bool, whether xc should be linked as a parameter

        """
        if k2 is None:
            super(Changepoint,self).__init__([k1],"changepoint")
            k2 = k1
        else:
            super(Changepoint,self).__init__([k1,k2],"changepoint")

        self.k1 = k1
        self.k2 = k2

        self.kc = Param('kc', kc, Logexp())
        self.link_parameter(self.kc)

        self.changepointParameter = changepointParameter
        self.xc = np.array(xc)
        if self.changepointParameter:
            self.xc = Param('xc', self.xc)
            self.link_parameter(self.xc)
            self.xc.gradient = [[0]]

        self.cpDim = cpDim

    def Kdiag(self,X):
        xside = X[:,self.cpDim] < self.xc[:,self.cpDim]

        K1 = self.k1.Kdiag(X)
        K2 = self.k2.Kdiag(X)

        n1 = self.k1.K(self.xc,self.xc)
        n2 = self.k2.K(self.xc,self.xc)

        G1 = self.k1.K(X,self.xc) / n1
        G2 = self.k2.K(X,self.xc) / n2

        return np.where(xside,K1 + G1*G1*(self.kc-n1),K2 + G2*G2*(self.kc-n2))

    def K(self,X,X2=None):

        if X2 is None:
            X2 = X

        K1 = self.k1.K(X,X2)
        K2 = self.k2.K(X,X2)

        n1 = self.k1.K(self.xc,self.xc)
        n2 = self.k2.K(self.xc,self.xc)

        G11 = self.k1.K(X,self.xc) / n1
        G12 = self.k1.K(X2,self.xc) / n1
        G21 = self.k2.K(X,self.xc) / n2
        G22 = self.k2.K(X2,self.xc) / n2

        x1side = X[:,self.cpDim] < self.xc[:,self.cpDim]
        x1side_2 = X[:,self.cpDim] > self.xc[:,self.cpDim]
        x2side = X2[:,self.cpDim] < self.xc[:,self.cpDim]
        x2side_2 = X2[:,self.cpDim] > self.xc[:,self.cpDim]
        x1side_cp = np.abs(X[:,self.cpDim] - self.xc[:,self.cpDim]) < CP_TOL
        x2side_cp = np.abs(X2[:,self.cpDim] - self.xc[:,self.cpDim]) < CP_TOL

        k = np.where(
                    # X, X2 on same side
                    np.outer(x1side,x2side),K1 + np.dot(G11,G12.T)*(self.kc-n1),
                         np.where(np.outer(x1side_2,x2side_2), K2 + np.dot(G21,G22.T)*(self.kc-n2),
                                # X, X2 on opposite sides
                                  np.where(np.outer(x1side,x2side_2), np.dot(G11,G22.T)*self.kc,
                                           np.where(np.outer(x1side_2,x2side), np.dot(G21,G12.T)*self.kc,
                                            # X or X2 is on the changepoint, but not the other
                                                np.where(np.outer(x1side_cp,x2side), self.k1.K(self.xc,X2),
                                                    np.where(np.outer(x1side_cp,x2side_2), self.k2.K(self.xc,X2),
                                                        np.where(np.outer(x1side,x2side_cp), self.k1.K(X,self.xc),
                                                            np.where(np.outer(x1side_2,x2side_cp), self.k2.K(X,self.xc),
                                                                # both are changepoints
                                                                self.kc)
                         )))))))

        # print k.shape

        return k

    def update_gradients_full(self, dL_dK, X, X2=None):
        """"""

        if X2 is None:
            X2 = X

        k = self.K(X,X2)*dL_dK

        x1side = X[:,self.cpDim] < self.xc[:,self.cpDim]
        x1side_2 = X[:,self.cpDim] > self.xc[:,self.cpDim]
        x2side = X2[:,self.cpDim] < self.xc[:,self.cpDim]
        x2side_2 = X2[:,self.cpDim] > self.xc[:,self.cpDim]

        n1 = self.k1.K(self.xc,self.xc)
        n2 = self.k2.K(self.xc,self.xc)

        G11 = self.k1.K(X,self.xc) / n1
        G12 = self.k1.K(X2,self.xc) / n1
        G21 = self.k2.K(X,self.xc) / n2
        G22 = self.k2.K(X2,self.xc) / n2

        # dL_dK1 = dL_dK if X,X2 < xc:
        self.k1.update_gradients_full(np.where(np.outer(x1side,x2side),dL_dK + 1/n1 * (2-G11-G12) - np.dot(G11,G12.T),0),X,X2)

        # dL_dK2 = dL_dK if X,X2 > xc:
        self.k2.update_gradients_full(np.where(np.outer(x1side_2,x2side_2),dL_dK+ 1/n2 * (2-G21-G22) - np.dot(G21,G22.T),0),X,X2)


        self.kc.gradient = np.sum(dL_dK*
                np.where( np.outer(x1side,x2side),np.dot(G11,G12.T),
                         np.where(np.outer(x1side_2,x2side_2), np.dot(G21,G22.T),
                                  np.where(np.outer(x1side,x2side_2), np.dot(G11,G22.T),
                                           np.where(np.outer(x1side_2,x2side), np.dot(G21,G12.T), 1
                         )))))

        if self.changepointParameter:
            self.xc.gradient = [[0]]
