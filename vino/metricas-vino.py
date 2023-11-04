from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import pandas as pd
from math import sqrt
import statistics
from scipy.stats import ttest_ind
from scipy.stats import t
from scipy.stats import mannwhitneyu


def read_csv_data(path):
    df = pd.read_csv(path)
    return df


vino_pred = read_csv_data('wine_valores_prescriptivos')

vino_true = read_csv_data('test_wine_final_data.csv')

r_mae = mean_absolute_error(vino_true.c6, vino_pred.c6)

r_maec3 = mean_absolute_error(vino_true.c8, vino_pred.c8)

r_mse_c6 = mean_squared_error(vino_true.c6, vino_pred.c6)

r_mse_c8 = mean_squared_error(vino_true.c8, vino_pred.c8)

print(r_mae)
print(r_maec3)
print(r_mse_c6)
print(r_mse_c8)

r_rmse_c6 = sqrt(mean_squared_error(vino_true.c6, vino_pred.c6)) 

r_rmse_c8 = sqrt(mean_squared_error(vino_true.c8, vino_pred.c8)) 

print(r_rmse_c6)
print(r_rmse_c8)

#####################################
mean_c6_pred = sum(vino_pred.c6) / 480
mean_c6_true = sum(vino_true.c6) / 480

print("The mean is =", mean_c6_pred)

print("The mean c6 true is =", mean_c6_true)

SD_c6_pred = statistics.stdev(vino_pred.c6)
SD_c6_true = statistics.stdev(vino_true.c6)

print("sd pred C6:", SD_c6_pred) 

print("sd true C6:", SD_c6_true) 


se1, se2 = SD_c6_true/sqrt(480), SD_c6_pred/sqrt(480)

sed = sqrt(se1**2 + se2**2)

t = (mean_c6_true - mean_c6_pred) / sed

print(t)


t_stat, p = ttest_ind(vino_true.c6, vino_pred.c6)

print(f't={t_stat}, p={p}')

############################################

print(mannwhitneyu(vino_true.c8, vino_pred.c8))

##################################### sd para vino c8
mean_c8_pred = sum(vino_pred.c8) / 480
mean_c8_true = sum(vino_true.c8) / 480

print("The mean is =", mean_c8_pred)

print("The mean c8 true is =", mean_c8_true)

SD_c8_pred = statistics.stdev(vino_pred.c8)
SD_c8_true = statistics.stdev(vino_true.c8)

print("sd pred C8:", SD_c8_pred) 

print("sd true C8:", SD_c8_true) 