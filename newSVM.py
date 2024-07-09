from sklearn.base import TransformerMixin,BaseEstimator
class NewSVM(TransformerMixin,BaseEstimator):

    def __init__(self,alpha0=0.4,a=0.6,r=100):
      self.alpha0=alpha0
      self.a=a
      self.r=r

    def fit(self, Xtrain, Ytrain, verbose=True):
        Xtrain=Xtrain.T
        Ytrain=Ytrain.T[np.newaxis, :]
        d, m = Xtrain.shape
        print('Xtrain.shape',Xtrain.shape )
        print('Ytrain.shape',Ytrain.shape )
        

        # define hyperparameter
        geval_limit = 1e6;
        maxitr = int(geval_limit)
        rbox = 1e7
        taubar_arr = [0, 1, 2, 5, 10] # delay
        time_limit = 10;
        lambda1 = 1
        lambda2 = 0.5

        # output variable
        maxidx = len(taubar_arr) + 2
        hinge = np.zeros((maxidx, maxitr))
        acc = np.zeros((maxidx, maxitr))
        geval = np.zeros((maxidx, maxitr))
        itr_flag = np.zeros((maxidx, time_limit))
        runtimes = np.zeros((len(taubar_arr)))
        results = pd.DataFrame({"taubar": taubar_arr, "niter": [0]*len(taubar_arr), "hinge": [0]*len(taubar_arr), "acc": [0]*len(taubar_arr), "time": [0]*len(taubar_arr), "TP": [0]*len(taubar_arr), "TN": [0]*len(taubar_arr), "FP": [0]*len(taubar_arr), "FN": [0]*len(taubar_arr)})

        for idx in range(len(taubar_arr)):

            tau_max = taubar_arr[idx]
            print("r = %d tau_bar = %d\n"%(self.r, tau_max))

            grad = np.zeros((1, d))
            self.w = np.zeros((d, 1))
            grad_num = 0

            st = time.time()
            # start the algorithm
            for k in range(maxitr + 1):

                # runtime-related script
                runtime = time.time() - st
                if runtime > time_limit:
                    break
                # ****************************** #
                itr_flag[idx, int(np.ceil(runtime)-1)] = k
                # ****************************** #

                alpha =  (self.alpha0/(k+1)) * ((1/(8+2*((tau_max+1)**2)))**(1/self.a))
                sumvec = np.zeros((d, 1))
                for i in range(m+2):
                    if (k % (tau_max + 1) == 0) | (tau_max == 0):
                        grad = np.zeros((1, d))
                        grad_num = grad_num + 1

                        if i + 1 <= m: # subgradient of generalized hinge loss
                            if Ytrain[:,i].dot(Xtrain[:,i].dot(self.w)) <= 0:
                                grad = -(self.r/m) * Ytrain[:,i].dot(Xtrain[:,i].reshape(1,-1))
                            elif (Ytrain[:,i].dot(Xtrain[:,i].dot(self.w)) > 0) & (Ytrain[:,i].dot(Xtrain[:,i].dot(self.w)) < 1):
                                grad = -(1/m) * Ytrain[:,i].dot(Xtrain[:,i].reshape(1,-1))
                            else:
                                grad = np.zeros((1, d))

                        elif i + 1 == m+1: # L1 norm:
                            for j in range(d):
                                if self.w[j] < 0:
                                    grad[:,j] = -1
                                elif self.w[j] > 0:
                                    grad[:,j] = 1
                                else:
                                    grad[:,j] = 0
                            grad = lambda1 * grad

                        elif i + 1 == m+2: # L2 norm
                            grad = self.w.T.dot(lambda2)

                    y_ik = self.w - alpha*grad.T.reshape(-1,1)

                    # solve the subproblem
                    x_ik = np.minimum(np.maximum(y_ik, -rbox), rbox)
                    sumvec = sumvec + x_ik

                w_next = sumvec/(m+2)

                hinge[idx, k+1] = self.general_hinge_loss(Xtrain, Ytrain, self.r)/m
                acc[idx, k+1] = (self.classify(Xtrain)==Ytrain).sum()/m
                geval[idx, k+1] = grad_num
                self.w = w_next
            evaltime = time.time() - st
            runtimes[idx] = evaltime
            print(" nitr = %d hinge = %.6f time = %.6f\n"%(k, hinge[idx, k],runtimes[idx]))
        return self

    def general_hinge_loss(self,X, Y, r):
        """
        Compute general hinge loss

        Parameters
        ----------
        w : A d x 1 matrix
        X : A d x m matrix
        Y : A 1 x m matrix
        r : a constant

        Returns
        -------
        sum_loss : A numeric value.

        """
        d, m = X.shape
        res = np.zeros((m,1))

        for i in range(m):
            val = Y[:,i].dot(X[:,i].dot(self.w))
            if val <= 0:
                res[i] = (1 - r*val)
            elif (val < 1) & (val > 0):
                res[i] = (1 - val)
            elif val >= 1:
                res[i] = 0

        sum_loss = sum(res)
        return sum_loss

    def predict(self, X):
        return self.w.T@X.T

    def classify(self,X):
        return np.where(self.predict(X.T) >= 0.0, 1, -1)