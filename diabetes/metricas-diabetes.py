from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import pandas as pd
from math import sqrt
import statistics
from scipy.stats import ttest_ind
from scipy.stats import t


def read_csv_data(path):
    df = pd.read_csv(path)
    return df


diabetes_pred = read_csv_data('diabetes_valores_prescriptivos.csv')

diabetes_true = read_csv_data('test_diabetes_final_data.csv')

r_mae = mean_absolute_error(diabetes_true.c2, diabetes_pred.c2)
print(r_mae)

r_maec3 = mean_absolute_error(diabetes_true.c3, diabetes_pred.c3)
print(r_maec3)


r_mse_c2 = mean_squared_error(diabetes_true.c2, diabetes_pred.c2)

r_mse_c3 = mean_squared_error(diabetes_true.c3, diabetes_pred.c3)

print(r_mse_c2)
print(r_mse_c3)

r_rmse_c2 = sqrt(mean_squared_error(diabetes_true.c2, diabetes_pred.c2)) 

r_rmse_c3 = sqrt(mean_squared_error(diabetes_true.c3, diabetes_pred.c3)) 

print(r_rmse_c2)
print(r_rmse_c3)


mean_c2_pred = sum(diabetes_pred.c2) / 300
mean_c2_true = sum(diabetes_true.c2) / 300

print("The mean c2 is =", mean_c2_pred)

print("The mean c2 true is =", mean_c2_true)

SD_c2_pred = statistics.stdev(diabetes_pred.c2)
SD_c2_true = statistics.stdev(diabetes_true.c2)

print("sd pred C2:", SD_c2_pred) 

print("sd true C2:", SD_c2_true) 


se1, se2 = SD_c2_true/sqrt(300), SD_c2_pred/sqrt(300)

sed = sqrt(se1**2 + se2**2)

t = (mean_c2_true - mean_c2_pred) / sed

print(t)
###################################################
mean_c3_pred = sum(diabetes_pred.c3) / 300
mean_c3_true = sum(diabetes_true.c3) / 300

print("The mean C3 is =", mean_c3_pred)

print("The mean C3 true is =", mean_c3_true)

SD_c3_pred = statistics.stdev(diabetes_pred.c3)
SD_c3_true = statistics.stdev(diabetes_true.c3)

print("sd pred C3:", SD_c3_pred) 

print("sd true C3:", SD_c3_true) 



se1_c3, se2_c3 = SD_c3_true/sqrt(300), SD_c3_pred/sqrt(300)

sed_c3 = sqrt(se1_c3**2 + se2_c3**2)

t_c3 = (mean_c3_true - mean_c3_pred) / sed_c3

print(t_c3)


t_stat_C2, p = ttest_ind(diabetes_true.c2, diabetes_pred.c2)

print(f't C2={t_stat_C2}, p={p}')


t_stat, p = ttest_ind(diabetes_true.c3, diabetes_pred.c3)

print(f't C3={t_stat}, p={p}')