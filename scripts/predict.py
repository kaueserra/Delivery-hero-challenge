import pickle
import config
#import lightgbm as lgb
from scripts.data_preprocess import data_preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score , classification_report, confusion_matrix
import pandas as pd
import pathlib


class predict():

    def __init__ (self):
        self.final_vars = config.final_var
        self.model = config.model
        self.label_data = config.label_data
        dt_p = data_preprocessing()
        self.ds = dt_p.prepare_data()
        self.path = str(pathlib.Path().absolute())

    def pred(self):

        x_test = self.ds[self.final_vars]

        ds1 = pd.read_csv(self.path + self.label_data)

        self.ds = pd.merge(x_test, ds1, how='left', on='customer_id')

        y_test = ds1['is_returning_customer']
        
        with open(self.path +  self.model, 'rb') as file:
            pickle_model = pickle.load(file)

        # Calculate the accuracy score and predict target values

        self.ds['pred'] = pickle_model.predict(x_test)

        print("LightGBM F1 micro Score -> ", round(f1_score(self.ds['pred'], y_test, average='micro') * 100), 2)

        print('---'*30)

        print("LightGBM Classification Report -> ")

        print(classification_report(self.ds['pred'], y_test) )

        print('---' * 30)

        print("LightGBM Confusion Matrix -> ")

        print(confusion_matrix(y_test,self.ds['pred']))

        self.ds = self.ds.reset_index()

        self.ds[['customer_id','is_returning_customer','pred']].to_csv('labeled_data_pred.csv')



