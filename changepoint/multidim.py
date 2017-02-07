from GPy.kern import Kern
from GPy.kern.src.kern import CombinationKernel
from GPy.core.parameterization import Param
from GPy.core.parameterization.transformations import Logexp
import numpy as np

CP_TOL = 1e-5

class Changepoint(CombinationKernel):
    """Kernel for a changepoint at position xc """

    def __init__(self,k1,k2=None,kc=None,xc=np.array([[0]]),cpDim=[0],changepointParameter=False, extra_dims=[], changepointCheck='all'):
        """
        arguments:
            k1, k2: GPy.kern.Kernel
            kc: float, covariance at the changepoint
            xc: np.array, position of changepoint(s)
            cpDim: int, dimension that changepoint exists on
            changepointParameter: bool, whether xc should be linked as a parameter

        """
        kerns = [k1,k2,kc]
        while None in kerns:
            kerns.remove(None)

        super(Changepoint,self).__init__(kerns,"changepoint", extra_dims=extra_dims)

        if k2 is None:
            # super(Changepoint,self).__init__([k1],"changepoint")
            k2 = k1
        if kc is None:
            kc = k1
        # else:
        #     super(Changepoint,self).__init__([k1,k2],"changepoint")

        self.k1 = k1
        self.k2 = k2
        self.kc = kc

        # self.kc = Param('kc', kc, Logexp())
        # self.link_parameter(self.kc)

        self.xc = np.array(xc)
        # assert self.xc.shape[1] == self.input_dim, 'must provide changepoint matching input dim'

        self.cpDim = cpDim
        # self.nonCpInd = range(self.input_dim+1)
        # [self.nonCpInd.remove(cpd) for cpd in self.cpDim]
        self.changepointCheck = changepointCheck

    def Kdiag(self,X):
        xside = X[:,self.cpDim] < self.xc[:,self.cpDim]

        K1 = self.k1.Kdiag(X)
        K2 = self.k2.Kdiag(X)

        n1 = self.k1.K(self.xc,self.xc)
        n2 = self.k2.K(self.xc,self.xc)

        G1 = self.k1.K(X,self.xc) / n1
        G2 = self.k2.K(X,self.xc) / n2

        return np.where(xside,K1 + G1*G1*(self.kc-n1),K2 + G2*G2*(self.kc-n2))

    def side(self,X):

        if self.changepointCheck == 'all':
            return np.array([np.all((x<self.xc)[:,self.cpDim]) for x in X])
        elif self.changepointCheck == 'match':
            select = []
            for i in range(X.shape[0]):
                select.append(np.where(np.all(self.xc[:,self.extra_dims]==X[i,self.extra_dims], 1))[0][0])

            return X[:,self.cpDim][:,0] < self.xc[select,self.cpDim]

    def K(self,X,X2=None):

        if X2 is None:
            X2 = X

        K1 = self.k1.K(X,X2)
        K2 = self.k2.K(X,X2)

        kc1 = self.k1.K(self.xc[:,self.cpDim], self.xc[:,self.cpDim])
        kc1_inv = np.linalg.inv(kc1)
        kc2 = self.k2.K(self.xc[:,self.cpDim], self.xc[:,self.cpDim])
        kc2_inv = np.linalg.inv(kc2)
        kcc = self.kc.K(self.xc[:,self.cpDim], self.xc[:,self.cpDim])

        x1side = self.side(X)
        x2side = self.side(X2)

        #####################
        ## Multiple XC for extra dim

        # G11 = np.dot(kc1_inv,self.k1.K(X,self.xc))
        # G12 = np.dot(kc1_inv,self.k1.K(X2,self.xc))
        # G21 = np.dot(kc2_inv,self.k2.K(X,self.xc))
        # G22 = np.dot(kc2_inv,self.k2.K(X2,self.xc))

        G11 = np.dot(self.k1.K(X,self.xc), kc1_inv)
        G12 = np.dot(self.k1.K(X2,self.xc), kc1_inv)
        G21 = np.dot(self.k2.K(X,self.xc), kc2_inv)
        G22 = np.dot(self.k2.K(X2,self.xc), kc2_inv)

        ######################
        ## Single xc

        # x1side = X[:,self.cpDim] < self.xc[:,self.cpDim]
        # x1side_2 = X[:,self.cpDim] > self.xc[:,self.cpDim]
        # x2side = X2[:,self.cpDim] < self.xc[:,self.cpDim]
        # x2side_2 = X2[:,self.cpDim] > self.xc[:,self.cpDim]
        # x1side_cp = np.abs(X[:,self.cpDim] - self.xc[:,self.cpDim]) < CP_TOL
        # x2side_cp = np.abs(X2[:,self.cpDim] - self.xc[:,self.cpDim]) < CP_TOL
        #
        # xc = np.zeros((X.shape[0], X.shape[1]))
        # xc2 = np.zeros((X2.shape[0],X2.shape[1]))
        # xc[:,self.cpDim] = self.xc
        # xc2[:,self.cpDim] = self.xc
        #
        # if self.input_dim > 1:
        #    ind = range(self.input_dim)
        #    ind.remove(self.cpDim)
        #    xc[:,ind] = X[:,ind]
        #    xc2[:,ind] = X2[:,ind]
        #
        # G11 = np.diag(self.k1.K(X,xc))[:,None] / self.k1.variance
        # G12 = np.diag(self.k1.K(X2,xc))[:,None] / self.k1.variance
        # G21 = np.diag(self.k2.K(X,xc))[:,None] / self.k2.variance
        # G22 = np.diag(self.k2.K(X2,xc))[:,None] / self.k2.variance

        # kc = self.kc*np.ones((self.xc.shape[0], self.xc.shape[0]))

        k = np.where(
                    # X, X2 on same side
                    np.outer(x1side==0,x2side==0),K1 + np.dot(G11,np.dot(kcc-kc1,G12.T)),
                         np.where(np.outer(x1side==1,x2side==1), K2 + np.dot(G21,np.dot(kcc-kc2,G22.T)),
                    # np.outer(x1side==0,x2side==0),K1 + np.dot(G11,G12.T)*(self.kc-kc1),
                    #      np.where(np.outer(x1side==1,x2side==1), K2 + np.dot(G21,G22.T)*(self.kc-kc2),
                                # X, X2 on opposite sides
                                  np.where(np.outer(x1side==0,x2side==1), np.dot(G11,np.dot(kcc,G22.T)),
                                           np.where(np.outer(x1side==1,x2side==0), np.dot(G21,np.dot(kcc,G12.T)),
                                # np.where(np.outer(x1side==0,x2side==1), np.dot(G11,G22.T)*self.kc,
                                #          np.where(np.outer(x1side==1,x2side==0), np.dot(G21,G12.T)*self.kc,
                                            # X or X2 is on the changepoint, but not the other
                                                # np.where(np.outer(x1side_cp,x2side), self.k1.K(self.xc,X2),
                                                #     np.where(np.outer(x1side_cp,x2side_2), self.k2.K(self.xc,X2),
                                                #         np.where(np.outer(x1side,x2side_cp), self.k1.K(X,self.xc),
                                                #             np.where(np.outer(x1side_2,x2side_cp), self.k2.K(X,self.xc),
                                                                # both are changepoints
                                                                self.kc)
                         )))#))))

        # print k.shape

        return k.astype(float)

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
