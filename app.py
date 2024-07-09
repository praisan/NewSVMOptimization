import json
import time
from flask import Flask, Response, render_template, request,flash, redirect, url_for,send_from_directory
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from forms import MyForm

from flask_uploads import configure_uploads, UploadSet, DATA
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from ydata_profiling import ProfileReport
import io
from plotly.io import to_html
import plotly.figure_factory as ff
import plotly.graph_objs as go
from summarytools import dfSummary
from joblib import dump, load
import os
import newSVM as sv
import threading


app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder to store uploaded files
app.config['UPLOAD_FOLDER_file'] = 'uploaded_file.csv'  # Folder to store uploaded files
app.config['MODEL_FOLDER'] = 'models'  # Folder to store model files
df=[]

# Flask-Uploads configuration
csvs = UploadSet('data', DATA)
app.config['UPLOADED_DATA_DEST'] = 'uploads'
configure_uploads(app, csvs)

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

@app.route('/form', methods=['GET', 'POST'])
def form():
    form = MyForm()
    if form.validate_on_submit():
        flash('Form submitted successfully')
        # Process the form data here if necessary
        return redirect(url_for('form'))
    return render_template('form.html', form=form)

@app.route('/', methods=['GET'])
def index():
    return render_template('index2.html')

@app.route('/upload-csv2', methods=['POST'])
def upload_csv2():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    header = request.form.get('header') == 'on'  # Check if the checkbox was ticked

    if file and file.filename.endswith('.csv'):
        try:
            if header:
                df = pd.read_csv(io.StringIO(file.stream.read().decode("UTF8")), sep=",")
            else:
                df = pd.read_csv(io.StringIO(file.stream.read().decode("UTF8")), sep=",", header=None)

            # EDA using summarytools
            eda_report = dfSummary(df)
            eda_report_html = eda_report.to_html()

            custom_html = '''
            <style>
                .fixed-next-button {
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    z-index: 1000;
                }
            </style>
            <div class="fixed-next-button">
                <a href="/next-page"><button>Next</button></a>
            </div>
            '''
            eda_report_html = eda_report_html.replace('</div>', f'</div>{custom_html}')

            return render_template('eda_dfSummary.html', eda_report=eda_report_html)
        except Exception as e:
            return render_template('error.html', error=str(e))

    else:
        return render_template('error.html', error='Unsupported file format')

@app.route('/upload-csv', methods=['POST'])
def upload_csv():
    global df
    # if 'file' not in request.files:
    #     return redirect(url_for('index'))
    if 'file' not in request.files:
        return render_template('error.html', error='No file part')
    

    file = request.files['file']
    header = request.form.get('header') == 'on'  # Check if the checkbox was ticked

    if file and file.filename.endswith('.csv'):
        try:
            # filepath = os.path.join(app.config['UPLOAD_FOLDER'], app.config['UPLOAD_FOLDER_file'])
            # file.save(filepath)
            if header:
                df = pd.read_csv(io.StringIO(file.stream.read().decode("UTF8")), sep=",")
            else:
                df = pd.read_csv(io.StringIO(file.stream.read().decode("UTF8")), sep=",", header=None)

            # df = pd.read_csv(filepath)
            profile = ProfileReport(df, explorative=True)
            profile.config.html.navbar_show=False
            profile_html = profile.to_html()
            return render_template('eda.html', eda_report=profile_html)
        except Exception as e:
            return render_template('error.html', error='Invalid CSV format and: '+str(e))

    else:
        return render_template('error.html', error='Unsupported file format')

@app.route('/training_results', methods=['POST'])
def next_page():
        # def train_and_get_results():
        global df

        # Split the data into train-test sets
        X = df.iloc[:, :-1].values.copy()
        y = df.iloc[:, -1].values.copy()
        split_proportion = float(request.form.get('split', 0.2)) 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_proportion, random_state=42)
        print('type(X_train): ',type(X_train))
        print('type(y_train): ',type(y_train))

        # Calculate train-test data information
        n_feature=X.shape[1]
        train_size = len(X_train)
        test_size = len(X_test)
        total_size = len(df)
        train_percent = (train_size / total_size) * 100
        test_percent = (test_size / total_size) * 100

        # Train the SVM model
        newSVM = NewSVM()
        newSVM.fit(X_train, y_train)

        model_directory = app.config['MODEL_FOLDER']
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        
        model_filename = 'new_svm_model.joblib'
        model_filepath = os.path.join(model_directory, model_filename)
        
        dump(newSVM, model_filepath)

        df_head_html = df.head().to_html(classes="table table-striped table-bordered")

        fig = go.Figure(data=[
            go.Bar(name='Train Set', y=[''], x=[train_size], orientation='h', marker_color='skyblue'),
            go.Bar(name='Evaluation Set', y=[''], x=[test_size], orientation='h', marker_color='lightcoral')
        ])

        fig.update_layout(
            barmode='stack',
            title_text='Train/Evaluation Split (Horizontal)',
            xaxis_title="Number of Samples",
            yaxis_title="Dataset",
        ) 
        data_split_plot_html = to_html(fig, full_html=False)  

        # Make predictions and calculate confusion matrix
        y_pred = newSVM.predict(X_test)
        y_pred_c = np.where(y_pred >= 0.0, 1, -1)
        # y_pred_proba = newSVM.predict_proba(X_test)
        cm = confusion_matrix(y_test, y_pred_c.T)  

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
        cm_plot_html = to_html(fig, full_html=False)

        # # Generate ROC Curve Plot
        # fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:,1])
        # # roc_auc = auc(fpr, tpr)
        # roc_fig = go.Figure()
        # roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
        # roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random'))
        # roc_fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
        # roc_plot_html = to_html(roc_fig, full_html=False)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[str(a) for a in newSVM.taubar_arr],
            y=newSVM.runtimes,
            name='Runtimes',
            marker_color='rgb(55, 83, 109)'))
        fig.update_layout(
            # title='Runtimes',
            xaxis_title='delay',
            yaxis_title='Time (seconds)')
        runtimes_plot_html = to_html(fig, full_html=False)

        fig = go.Figure()
        # Add traces for each row in the acc array
        for i in range(len(newSVM.taubar_arr)):
            hinge=newSVM.hinge[i,newSVM.hinge[i,:]>0]
            fig.add_trace(go.Scatter(x=list(range(hinge.shape[0])), y=hinge, name=f"delay {newSVM.taubar_arr[i]}"))
        fig.update_layout(
            # title="hinge for each row in the acc array",
            xaxis_title="Iteration",
            yaxis_title="hinge",)
        hinge_plot_html = to_html(fig, full_html=False)

        fig = go.Figure()
        for i in range(len(newSVM.taubar_arr)):
            acc=newSVM.acc[i,newSVM.acc[i,:]>0]
            fig.add_trace(go.Scatter(x=list(range(acc.shape[0])), y=acc, name=f"delay {newSVM.taubar_arr[i]}"))
        fig.update_layout(
            # title="Accuracy for each row in the acc array",
            xaxis_title="Iteration",
            yaxis_title="Accuracy",)
        accuracy_plot_html = to_html(fig, full_html=False)

        # Render the confusion matrix on a new page (can also convert cm to HTML table)
        # return render_template('results.html', confusion_matrix_plot=cm_plot_html, roc_curve_plot=roc_plot_html)
        return render_template('results.html', 
                            confusion_matrix_plot=cm_plot_html,
                            runtimes_plot=runtimes_plot_html, 
                            training_hinge_plot=hinge_plot_html,
                            training_accuracy_plot=accuracy_plot_html,
                            data_split_plot=data_split_plot_html,
                            df_head=df_head_html,
                            #    roc_curve_plot=roc_plot_html,
                            train_size=train_size, 
                            test_size=test_size,
                            total_size=train_size+test_size,
                            n_feature=n_feature,
                            train_percent=train_percent, 
                            test_percent=test_percent,
                            model_filename=model_filename)
        # threading.Thread(target=train_and_get_results).start()
        # return render_template('training_in_progress.html')


@app.route('/download-model/<filename>', methods=['GET'])
def download_model(filename):
    # Make sure the directory is correct and the file exists
    directory = app.config['MODEL_FOLDER']
    # return send_from_directory("..\\..\\"+directory, filename, as_attachment=True)
    print("..\\"+directory)
    return send_from_directory("..\\"+directory, filename, as_attachment=True)

# @app.route('/form_upload', methods=['GET', 'POST'])
# def form_upload():
#     form = UploadForm()
    
#     if form.validate_on_submit():
#         filename = csvs.save(form.file.data)
#         filepath = csvs.path(filename)
#         print('I am here 2')
        
#         # Load CSV
#         data = pd.read_csv(filepath)
#         # Assuming last column as target, rest as features
#         X = data.iloc[:, :-1]
#         y = data.iloc[:, -1]

#         # Splitting the dataset
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         # Training SVM
#         clf = SVC()
#         clf.fit(X_train, y_train)
        
#         # Optionally: Save the trained model
#         # joblib.dump(clf, 'svm_model.pkl')
        
#         flash('Model trained successfully!')
#         return redirect(url_for('form_upload'))

#     return render_template('upload.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)