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


estudiante_pred = read_csv_data('re_valores_prescriptivos.csv')

estudiante_true = read_csv_data('test_student_final_data.csv')

r_mae = mean_absolute_error(estudiante_true.c9, estudiante_pred.c9)
print(r_mae)

r_maec10 = mean_absolute_error(estudiante_true.c10, estudiante_pred.c10)
print(r_maec10)


r_maec11 = mean_absolute_error(estudiante_true.c11, estudiante_pred.c11)
print(r_maec11)

r_maec12 = mean_absolute_error(estudiante_true.c12, estudiante_pred.c12)
print(r_maec12)

r_mse_c9 = mean_squared_error(estudiante_true.c9, estudiante_pred.c9)

r_mse_c10= mean_squared_error(estudiante_true.c10, estudiante_pred.c10)

r_mse_c11= mean_squared_error(estudiante_true.c11, estudiante_pred.c11)

r_mse_c12= mean_squared_error(estudiante_true.c12, estudiante_pred.c12)

print(r_mse_c9)
print(r_mse_c10)

print(r_mse_c11)
print(r_mse_c12)


r_rmse_c9 = sqrt(mean_squared_error(estudiante_true.c9, estudiante_pred.c9)) 

r_rmse_c10 = sqrt(mean_squared_error(estudiante_true.c10, estudiante_pred.c10))

r_rmse_c11 = sqrt(mean_squared_error(estudiante_true.c11, estudiante_pred.c11)) 

r_rmse_c12 = sqrt(mean_squared_error(estudiante_true.c12, estudiante_pred.c12)) 

print(r_rmse_c9)
print(r_rmse_c10)

print(r_rmse_c11)
print(r_rmse_c12)


#####################################
mean_c9_pred = sum(estudiante_pred.c9) / 212
mean_c9_true = sum(estudiante_true.c9) / 212

print("The mean is =", mean_c9_pred)

print("The mean C9 true is =", mean_c9_true)

SD_c9_pred = statistics.stdev(estudiante_pred.c9)
SD_c9_true = statistics.stdev(estudiante_true.c9)

print("sd pred C9:", SD_c9_pred) 

print("sd true C9:", SD_c9_true) 


se1, se2 = SD_c9_true/sqrt(212), SD_c9_pred/sqrt(212)

sed = sqrt(se1**2 + se2**2)

t = (mean_c9_true - mean_c9_pred) / sed

print(t)


t_stat, p = ttest_ind(estudiante_true.c9, estudiante_pred.c9)

print(f't={t_stat}, p C9={p}')


mean_c10_pred = sum(estudiante_pred.c10) / 212
mean_c10_true = sum(estudiante_true.c10) / 212

print("The mean is =", mean_c10_pred)

print("The mean is =", mean_c10_true)

SD_c10_pred = statistics.stdev(estudiante_pred.c10)
SD_c10_true = statistics.stdev(estudiante_true.c10)

print("sd pred C2:", SD_c10_pred) 

print("sd true C2:", SD_c10_true) 


sec10_1, sec10_2 = SD_c10_true/sqrt(212), SD_c10_pred/sqrt(212)

sedc10 = sqrt(sec10_1**2 + sec10_2**2)

t = (mean_c10_true - mean_c10_pred) / sedc10

print(t)


t_stat, p = ttest_ind(estudiante_true.c10, estudiante_pred.c10)

print(f't={t_stat}, p C10={p}')



mean_c10_pred = sum(estudiante_pred.c10) / 212
mean_c10_true = sum(estudiante_true.c10) / 212

print("The mean is =", mean_c10_pred)

print("The mean true is =", mean_c10_true)

SD_c10_pred = statistics.stdev(estudiante_pred.c10)
SD_c10_true = statistics.stdev(estudiante_true.c10)

print("sd pred C10:", SD_c10_pred) 

print("sd true C10:", SD_c10_true) 


sec10_1, sec10_2 = SD_c10_true/sqrt(212), SD_c10_pred/sqrt(212)

sedc10 = sqrt(sec10_1**2 + sec10_2**2)

t = (mean_c10_true - mean_c10_pred) / sedc10

print(t)


t_stat, p = ttest_ind(estudiante_true.c10, estudiante_pred.c10)

print(f't={t_stat}, p C10={p}')


#####################################
mean_c11_pred = sum(estudiante_pred.c11) / 212
mean_c11_true = sum(estudiante_true.c11) / 212

print("The mean is =", mean_c11_pred)

print("The mean true is =", mean_c11_true)

SD_c11_pred = statistics.stdev(estudiante_pred.c11)
SD_c11_true = statistics.stdev(estudiante_true.c11)

print("sd pred C11:", SD_c11_pred) 

print("sd true C11:", SD_c11_true) 


sec11_1, sec11_2 = SD_c11_true/sqrt(212), SD_c11_pred/sqrt(212)

sedc11 = sqrt(sec11_1**2 + sec11_2**2)

t = (mean_c11_true - mean_c11_pred) / sedc11

print(t)


t_stat, p = ttest_ind(estudiante_true.c11, estudiante_pred.c11)

print(f't={t_stat}, p C11={p}')


#####################################
mean_c12_pred = sum(estudiante_pred.c12) / 212
mean_c12_true = sum(estudiante_true.c12) / 212

print("The mean is =", mean_c12_pred)

print("The mean true is =", mean_c12_true)

SD_c12_pred = statistics.stdev(estudiante_pred.c12)
SD_c12_true = statistics.stdev(estudiante_true.c12)

print("sd pred C12:", SD_c12_pred) 

print("sd true C12:", SD_c12_true) 


sec12_1, sec12_2 = SD_c12_true/sqrt(212), SD_c12_pred/sqrt(212)

sedc12 = sqrt(sec12_1**2 + sec12_2**2)

t = (mean_c12_true - mean_c12_pred) / sedc12

print(t)


t_stat, p = ttest_ind(estudiante_true.c12, estudiante_pred.c12)

print(f't={t_stat}, p C12={p}')