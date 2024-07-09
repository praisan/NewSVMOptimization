import time
from flask import Flask, render_template, request,flash, redirect, url_for,send_from_directory
import numpy as np
from forms import MyForm

from flask_uploads import configure_uploads, UploadSet, DATA
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,roc_curve
from ydata_profiling import ProfileReport
import io
from plotly.io import to_html
import plotly.graph_objs as go
from summarytools import dfSummary
from joblib import dump
import os
from newSVM import NewSVM, accuracy_plot,data_split_plot,cm_plot, hinge_plot, roc_plot, runtimes_plot



app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder to store uploaded files
app.config['UPLOAD_FOLDER_file'] = 'uploaded_file.csv'  # Folder to store uploaded files
app.config['MODEL_FOLDER'] = 'models'  # Folder to store model files
df=[]

# Flask-Uploads configuration
csvs = UploadSet('data', DATA)
app.config['UPLOADED_DATA_DEST'] = 'uploads'
configure_uploads(app, csvs)


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
        data_split_plot_html = to_html(data_split_plot(train_size,test_size),full_html=False)

        y_pred = newSVM.predict(X_test)
        y_pred_c = np.where(y_pred >= 0.0, 1, -1)
        # y_pred_proba = newSVM.predict_proba(X_test)
        cm = confusion_matrix(y_test, y_pred_c.T)  

        cm_plot_html=to_html(cm_plot(cm),full_html=False) 

        # # Generate ROC Curve Plot
        # fpr, tpr, _ = roc_curve(y_test, y_pred.T)        
        # roc_plot_html = to_html(roc_plot(fpr,tpr), full_html=False)
        runtimes_plot_html = to_html(runtimes_plot(newSVM), full_html=False)        
        hinge_plot_html = to_html(hinge_plot(newSVM), full_html=False)
        accuracy_plot_html = to_html(accuracy_plot(newSVM), full_html=False)

        return render_template('results.html', 
                            confusion_matrix_plot=cm_plot_html,
                            runtimes_plot=runtimes_plot_html, 
                            training_hinge_plot=hinge_plot_html,
                            training_accuracy_plot=accuracy_plot_html,
                            data_split_plot=data_split_plot_html,
                            df_head=df_head_html,
                            # roc_curve_plot=roc_plot_html,
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