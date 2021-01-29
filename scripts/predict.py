import pickle
import config
import lightgbm as lgb
from scripts.data_preprocess import data_preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score , classification_report, confusion_matrix
import pandas as pd
import pathlib


class predict():

    def __init__ (self):
        self.final_vars = config.final_var
        self.final_var_transf = config.final_var_transf
        self.model = config.model
        self.label_data = config.label_data
        self.dt_p = data_preprocessing()
        self.path = str(pathlib.Path().absolute())


    def pred(self):

        final_ds = self.dt_p.prepare_data()

        ds1 = pd.read_csv(self.path + self.label_data)

        final_ds = pd.merge(final_ds, ds1, how='left', left_index=True , right_on='customer_id')

        y_test = final_ds['is_returning_customer']

        x_test = final_ds[self.final_var_transf]
        
        with open(self.path +  self.model, 'rb') as file:
            pickle_model = pickle.load(file)

        # Calculate the accuracy score and predict target values

        final_ds['pred'] = pickle_model.predict(x_test)

        print("LightGBM F1 micro Score -> ", round(f1_score(final_ds['pred'], y_test, average='micro') * 100), 2)

        print('---'*30)

        print("LightGBM Classification Report -> ")

        print(classification_report(final_ds['pred'], y_test) )

        print('---' * 30)

        print("LightGBM Confusion Matrix -> ")

        print(confusion_matrix(y_test,final_ds['pred']))

        final_ds = final_ds.reset_index()

        final_ds[['customer_id','is_returning_customer','pred']].to_csv('labeled_data_pred.csv')



