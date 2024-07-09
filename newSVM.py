import time
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin,BaseEstimator
from plotly.io import to_html
import plotly.graph_objs as go

class NewSVM(TransformerMixin,BaseEstimator):

    def __init__(self,alpha0=0.4,a=0.6,r=100):
      self.alpha0=alpha0
      self.a=a
      self.r=r

    def fit(self, Xtrain, Ytrain, verbose=True,progress_callback=None):
        Xtrain=Xtrain.T
        Ytrain=Ytrain.T[np.newaxis, :]
        d, m = Xtrain.shape
        if verbose:
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
        maxidx = len(taubar_arr)
        hinge = np.zeros((maxidx, maxitr))
        acc = np.zeros((maxidx, maxitr))
        # geval = np.zeros((maxidx, maxitr))
        itr_flag = np.zeros((maxidx, time_limit))
        runtimes = np.zeros((maxidx))
        # results = pd.DataFrame({"taubar": taubar_arr, "niter": [0]*maxidx, "hinge": [0]*maxidx, "acc": [0]*maxidx, "time": [0]*maxidx, "TP": [0]*maxidx, "TN": [0]*maxidx, "FP": [0]*maxidx, "FN": [0]*maxidx})

        for idx in range(maxidx):

            tau_max = taubar_arr[idx]
            if verbose: print("r = %d tau_bar = %d\n"%(self.r, tau_max))

            grad = np.zeros((1, d))
            self.w = np.zeros((d, 1))
            # grad_num = 0

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
                        # grad_num = grad_num + 1

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
                # geval[idx, k+1] = grad_num
                self.w = w_next
            evaltime = time.time() - st
            runtimes[idx] = evaltime
            if verbose: print(" nitr = %d hinge = %.6f time = %.6f\n"%(k, hinge[idx, k],runtimes[idx]))
            if progress_callback:  # Check if a callback is provided
                progress = (idx*maxitr+k) / (maxidx*maxitr)  
                progress_callback(progress)  # Call the callback
        self.hinge=hinge
        self.acc=acc
        self.runtimes=runtimes
        self.taubar_arr=taubar_arr
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
    
def data_split_plot(train_size,test_size):
    fig = go.Figure(data=[
        go.Bar(name='Train Set', y=[''], x=[train_size], orientation='h', marker_color='skyblue'),
        go.Bar(name='Evaluation Set', y=[''], x=[test_size], orientation='h', marker_color='lightcoral')
    ])

    fig.update_layout(
        barmode='stack',
        title_text='Train/Evaluation Split',
        xaxis_title="Number of Samples",
        yaxis_title="Dataset",
    ) 
    return fig

def cm_plot(cm):
    labels = ["Class 0", "Class 1"]
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels,
        hoverongaps=False, colorscale="Blues"  
        ))
    for i in range(len(cm)):
        for j in range(len(cm)):
            fig.add_annotation(
                x=labels[i], y=labels[j],
                text=str(cm[i][j]),
                showarrow=False,
                font=dict(color="white" if cm[i][j] > np.max(cm)/2 else "black")  # Set text color based on cell value
    )
            fig.update_layout(
                xaxis_title="Predicted Label",
                yaxis_title="True Label",
                xaxis=dict(tickmode='linear'), 
                yaxis=dict(tickmode='linear'))
    return fig

def roc_plot(fpr,tpr):
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random'))
    roc_fig.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    return roc_fig

def runtimes_plot(newSVM):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[str(a) for a in newSVM.taubar_arr],
        y=newSVM.runtimes,
        name='Runtimes',
        marker_color='rgb(55, 83, 109)'))
    fig.update_layout(xaxis_title='delay', yaxis_title='Time (seconds)')
    return fig

def hinge_plot(newSVM):
    fig = go.Figure()
    for i in range(len(newSVM.taubar_arr)):
        hinge=newSVM.hinge[i,newSVM.hinge[i,:]>0]
        fig.add_trace(go.Scatter(x=list(range(hinge.shape[0])), y=hinge, name=f"delay {newSVM.taubar_arr[i]}"))
    fig.update_layout(xaxis_title="Iteration", yaxis_title="hinge",)
    return fig

def accuracy_plot(newSVM):
    fig = go.Figure()
    for i in range(len(newSVM.taubar_arr)):
        acc=newSVM.acc[i,newSVM.acc[i,:]>0]
        fig.add_trace(go.Scatter(x=list(range(acc.shape[0])), y=acc, name=f"delay {newSVM.taubar_arr[i]}"))
    fig.update_layout(xaxis_title="Iteration", yaxis_title="Accuracy",)
    return fig