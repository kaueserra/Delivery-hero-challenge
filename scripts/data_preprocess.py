import config
import helpers
import pandas as pd
import numpy as np
from datetime import datetime
import dateutil.relativedelta
from sklearn.preprocessing import minmax_scale
import pathlib
import os

class data_preprocessing():

    def __init__ (self):
        self.final_var = config.final_var
        self.label_data = config.label_data
        self.predictors_data = config.predictors_data
        self.col_name1 = config.col_name1
        self.col_name2 = config.col_name2
        self.today = config.today
        self.cat_var =config.cat_var
        self.path = str(pathlib.Path().absolute())

    def prepare_data(self):

        # get the data
        ds1 = pd.read_csv(self.path + self.label_data)
        ds2 = pd.read_csv(self.path + self.predictors_data)

        self.ds = pd.merge(ds1, ds2, how='left', on='customer_id')

        # basic data transformation and sort by date

        self.ds['order_date'] = pd.to_datetime(self.ds['order_date'], format="%Y-%m-%d")

        self.ds['day_time_transaction'] = self.ds['order_hour'].apply(lambda x: helpers.day_time_func(x))

        self.ds['voucher_bol'] = np.where(self.ds['voucher_amount'] > 0, 1, 0)

        self.ds['delivery_free'] = np.where(self.ds['delivery_fee'] > 0, 1, 0)

        self.ds = self.ds.sort_values(by=['order_date', 'customer_id'], ascending=False)

        # Only the selected aggregations

        ds_cust1 = pd.DataFrame(self.ds.groupby('customer_id').agg({
            'customer_id': 'count',
            'order_date': 'first',
            'day_time_transaction': lambda x: x.nunique(),
            'is_failed': 'mean',
            'voucher_amount': 'mean',
            'delivery_fee': 'sum',
            'amount_paid': 'sum',
            'restaurant_id': lambda x: x.nunique(),
            'city_id': lambda x: x.nunique(),
            'payment_id': lambda x: x.nunique(),
            'platform_id': lambda x: x.nunique(),
            'transmission_id': lambda x: x.nunique(),
            'voucher_bol': 'mean',
            'delivery_free': 'mean', }))

        ds_cust1.columns = self.col_name1

        last_month_ds = self.ds[self.ds.order_date > (self.today - dateutil.relativedelta.relativedelta(months=12))]

        ds_cust2 = pd.DataFrame(last_month_ds.groupby('customer_id').agg({
            'order_date': 'last',
            'is_failed': 'sum',
            'amount_paid': 'sum'}))

        ds_cust2.columns = self.col_name2

        last_month_ds = self.ds[self.ds.order_date > (self.today - dateutil.relativedelta.relativedelta(months=6))]

        ds_cust3 = pd.DataFrame(last_month_ds.groupby('customer_id').agg({
            'voucher_amount': 'sum',}))

        ds_cust3.columns = ['voucher_amount_sum_m6']

        # Merging and creating more features

        ds_agg_cust = pd.merge(ds_cust1, ds_cust2, how='left', left_index=True, right_index=True)

        ds_agg_cust = pd.merge(ds_agg_cust, ds_cust3, how='left', left_index=True, right_index=True)

        ds_agg_cust['fist_last_order_diff'] = (ds_agg_cust['last_order'] - ds_agg_cust['first_order']).dt.days

        ds_agg_cust['fist_order_today_diff'] = (self.today - ds_agg_cust['first_order']).dt.days

        ds_agg_cust['last_order_today_diff'] = (self.today - ds_agg_cust['last_order']).dt.days

        ds_agg_cust['month_first_order'] = ds_agg_cust.first_order.dt.month

        ds_agg_cust['month_last_order'] = ds_agg_cust.last_order.dt.month

        ds_agg_cust['month_last_order'] = (ds_agg_cust.last_order.dt.month).astype('Int64')

        ds_agg_cust['less_one_month_order'] = np.where(ds_agg_cust['last_order_today_diff'] < 31, 1, 0)

        ds_agg_cust['total_transac_bigger_5'] = np.where(ds_agg_cust['total_orders'] > 5, 1, 0)

        ds_all_raw = ds_agg_cust.copy()

        # Dropping date variables

        drop_var = list(ds_all_raw.select_dtypes(include=['datetime64']).columns)

        ds_all_raw = ds_all_raw.drop(drop_var, axis=1)

        # Using min-max scale on the numeric

        num_vars = list(set(list(self.final_var)) - set(self.cat_var))

        final_ds1 = pd.DataFrame(minmax_scale(ds_all_raw[num_vars]), index=ds_all_raw.index, columns=num_vars)

        # creating variable dummies for the categorical

        ds_all_raw[self.cat_var] = ds_all_raw[self.cat_var].apply(lambda x: x.astype('category'))

        final_ds2 = pd.get_dummies(ds_all_raw[self.cat_var], drop_first=True)

        final_ds = pd.merge(final_ds1, final_ds2, how='left', left_index=True, right_index=True) 

        return final_ds