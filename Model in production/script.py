#importing libraries
import os
import numpy as np
import flask
import pickle
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline, FeatureUnion

from flask import Flask, render_template, request

#creating instance of the class
app=Flask(__name__)

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')
    #return "Hello World"

#prediction function
def ValuePredictor(to_predict_list):
    to_predict = to_predict_list
    loaded_model = pickle.load(open("model.pkl","rb"))
    y_pred = np.where(loaded_model.predict_proba(to_predict)[:,1]>0.3,1,0)
    return y_pred[0]
columns = ['slope_of_peak_exercise_st_segment',
 'thal',
 'resting_blood_pressure',
 'chest_pain_type',
 'num_major_vessels',
 'fasting_blood_sugar_gt_120_mg_per_dl',
 'resting_ekg_results',
 'serum_cholesterol_mg_per_dl',
 'oldpeak_eq_st_depression',
 'sex',
 'age',
 'max_heart_rate_achieved',
 'exercise_induced_angina']

@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        #to_predict_list=list(to_predict_list.values())
        to_predict_list = list(to_predict_list.values())
        to_predict_list=pd.DataFrame(to_predict_list).T
        to_predict_list.columns = columns
        to_predict_list["chest_pain_type"] = int(to_predict_list["chest_pain_type"])
        to_predict_list["num_major_vessels"] = int(to_predict_list["num_major_vessels"])
        to_predict_list["resting_ekg_results"] = int(to_predict_list["resting_ekg_results"])
        result = ValuePredictor(to_predict_list)
        
        if int(result)==1:
            prediction='Heart Disease Present! Begin the treatment!'
        else:
            prediction='Heart Disease Not Present. No need to worry!'
            
        return render_template("result.html",prediction=prediction)

# data preprocessing
class ColumnSelectTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X[self.columns]
    
class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.labelers = {col: LabelEncoder().fit(X[col]) for col in X}
        return self
    
    def transform(self, X):
        return pd.DataFrame({col: self.labelers[col].transform(X[col])
                            for col in X})

numerical_features = ['slope_of_peak_exercise_st_segment', 
                      'resting_blood_pressure', 
                      'num_major_vessels',
                      'fasting_blood_sugar_gt_120_mg_per_dl',
                      'serum_cholesterol_mg_per_dl',
                      'oldpeak_eq_st_depression',
                      'age',
                      'max_heart_rate_achieved']

categorical_features = ['thal',
                        'chest_pain_type', 
                        'resting_ekg_results']

binary_features = ['sex',
                   'exercise_induced_angina']
    
# For categorical features
cat_pipe = Pipeline([
    ('cst', ColumnSelectTransformer(categorical_features)),
    ('cle', CustomLabelEncoder()),
    ('ohe', OneHotEncoder(sparse=False))
])

# For features we don't want to transform
passthrough_pipe = Pipeline([
    ('cst', ColumnSelectTransformer(numerical_features + binary_features))
])

# combining the above pipelines
feat_u = FeatureUnion([
    ('cat_pipe', cat_pipe),
    ('passthrough_pipe', passthrough_pipe)
])


if __name__ == "__main__":
	app.run(debug=True)
