import csv
import numpy as np


# create empty variable lists
temp, lat, long, elev = [], [], [], []


#extract data using csv library, more basic than pandas approach we use in chem courses
with open('temperature_data.csv', 'r') as file:
    csv_reader = csv.reader(file) # skip header
    next(csv_reader) # read each column and add values to the variable lists as float or int
    for row in csv_reader:
        try:
            current_temp = float(row[5])
        except ValueError:
            continue
        lat.append(float(row[2]))
        long.append(float(row[3]))
        elev.append(float(row[4]))
        temp.append(current_temp)


one = np.ones(len(temp))
matrix = np.array([one, lat, long, elev])     # matrix with shape (4, n)
y = np.array(temp)


beta = np.linalg.inv(matrix @ matrix.T) @ matrix @ y #beta = np.array([B_0, B_1, B_2, B_3])
B_0 = beta[0]
B_1 = beta[1]
B_2 = beta[2]
B_3 = beta[3]

print(f'Beta 0 is intercept: {B_0:.4f}')
print(f'Beta 1 is slope of latitude: {B_1:.4f}')
print(f'Beta 2 is slope of longtitude: {B_2:.4f}')
print(f'Beta 3 is slope of elevation: {B_3:.4f}')

beta_lstsq, *_ = np.linalg.lstsq(matrix.T, y, rcond=None)
print('\nValidation:')
print(f'  max |beta - beta_lstsq| : {np.max(np.abs(beta - beta_lstsq)):.2e}')
print(f'  solutions match         : {np.allclose(beta, beta_lstsq)}')
#  Large condition number => direct inverse is fragile, hence the lstsq check.
print(f'  condition number of XXᵀ : {np.linalg.cond(matrix @ matrix.T):.2e}')

with open('temperature_data.csv', mode='r', encoding='utf-8') as infile, \
     open('predicted_temperature_data.csv', mode='w', encoding='utf-8',newline='') as outfile:
     csv_reader = csv.reader(infile)
     csv_writer = csv.writer(outfile)

     header = next(csv_reader)
     header.append('predicted temperature')
     csv_writer.writerow(header)

     for row in csv_reader:
            lat = (float(row[2]))
            long = (float(row[3]))
            elev = (float(row[4]))
            temp_pre = B_0 + B_1 * lat + B_2 * long + B_3 * elev
            row.append(round(temp_pre,1))
            csv_writer.writerow(row)