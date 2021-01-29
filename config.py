from datetime import datetime
import os
import pathlib

path = os.path.abspath(os.getcwd())

final_var = ['fist_order_today_diff', 'total_transac_bigger_5', 'fist_last_order_diff', 'amount_paid_sum',
             'month_first_order', 'total_diff_time', 'total_rest', 'total_transmission', 'total_plat',
             'last_order_today_diff', 'total_pay', 'delivery_fee_sum', 'failed_mean', 'month_last_order',
             'failed_sum_m12', 'less_one_month_order', 'delivery_free_mean', 'total_city',
             'voucher_amount_sum_m6', 'voucher_bol_mean', 'amount_paid_m12']

col_name1 = ['total_orders', 'first_order', 'total_diff_time', 'failed_mean',
             'voucher_amount_mean', 'delivery_fee_sum', 'amount_paid_sum',
             'total_rest', 'total_city', 'total_pay', 'total_plat',
             'total_transmission', 'voucher_bol_mean', 'delivery_free_mean']

col_name2 = ['last_order', 'failed_sum_m12', 'amount_paid_m12']

today = datetime.strptime('2017-02-28', '%Y-%m-%d')

model = '/artifacts/final_model.pkl'

final_var_transf = ['total_rest', 'failed_sum_m12', 'delivery_fee_sum',
                    'fist_last_order_diff', 'amount_paid_sum', 'fist_order_today_diff',
                    'last_order_today_diff', 'voucher_amount_sum_m6', 'failed_mean',
                    'voucher_bol_mean', 'delivery_free_mean', 'amount_paid_m12',
                    'total_pay_2', 'total_pay_3', 'total_pay_4', 'total_pay_5',
                    'month_last_order_2', 'month_last_order_3', 'month_last_order_4',
                    'month_last_order_5', 'month_last_order_6', 'month_last_order_7',
                    'month_last_order_8', 'month_last_order_9', 'month_last_order_10',
                    'month_last_order_11', 'month_last_order_12',
                    'total_transac_bigger_5_1', 'total_plat_2', 'total_plat_3',
                    'total_plat_4', 'total_plat_5', 'total_plat_6', 'total_plat_7',
                    'less_one_month_order_1', 'total_transmission_2',
                    'total_transmission_3', 'total_transmission_4', 'total_transmission_5',
                    'total_transmission_6', 'total_transmission_7', 'total_city_2',
                    'total_city_3', 'total_city_4', 'total_city_5', 'total_city_6',
                    'total_city_7', 'total_city_8', 'total_city_9', 'total_city_10',
                    'total_city_11', 'total_city_12', 'total_city_20', 'total_city_21',
                    'total_city_24', 'total_city_26', 'total_diff_time_2',
                    'total_diff_time_3', 'total_diff_time_4', 'month_first_order_2',
                    'month_first_order_3', 'month_first_order_4', 'month_first_order_5',
                    'month_first_order_6', 'month_first_order_7', 'month_first_order_8',
                    'month_first_order_9', 'month_first_order_10', 'month_first_order_11',
                    'month_first_order_12']

label_data = '/data/machine_learning_challenge_labeled_data.csv'

predictors_data = '\data\machine_learning_challenge_order_data.csv'

cat_var = ['total_transmission', 'month_last_order', 'total_city', 'total_pay', 'month_first_order',
           'less_one_month_order', 'total_diff_time', 'total_plat', 'total_transac_bigger_5']
