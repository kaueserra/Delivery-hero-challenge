import datetime

final_var = ['fist_order_today_diff', 'total_transac_bigger_5', 'fist_last_order_diff', 'amount_paid_sum',
             'month_first_order','total_diff_time', 'total_rest', 'total_transmission', 'total_plat',
             'last_order_today_diff', 'total_pay', 'delivery_fee_sum', 'failed_mean', 'month_last_order',
             'failed_sum_m12', 'less_one_month_order','delivery_free_mean', 'total_city',
             'voucher_amount_sum_m6', 'voucher_bol_mean', 'amount_paid_m12']

col_name1 =  ['total_orders','first_order','total_diff_time','failed_mean' ,
              'voucher_amount_mean','delivery_fee_sum','amount_paid_sum',
              'total_rest','total_city','total_pay','total_plat',
              'total_transmission','voucher_bol_mean','delivery_free_mean']


col_name2 = ['last_order','failed_sum_m12','amount_paid_m12']

today = datetime.strptime('2017-02-28', '%Y-%m-%d')


model = 'artifacts/final_model.pkl'


label_data = 'data/machine_learning_challenge_labeled_data.csv'

predictors_data = 'data/machine_learning_challenge_order_data.csv'

