from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from flask_wtf.file import FileField, FileRequired, FileAllowed

class MyForm(FlaskForm):
    input1 = StringField('Input 1', validators=[DataRequired()])
    input2 = StringField('Input 2', validators=[DataRequired()])
    # ... repeat for other inputs up to input10
    input10 = StringField('Input 10', validators=[DataRequired()])
    submit = SubmitField('Submit')

class UploadForm(FlaskForm):
    file = FileField('CSV File', validators=[FileRequired(), FileAllowed(['csv'], 'CSV files only!')])
    submit = SubmitField('Upload and Train')