
import csv
import numpy as np

def data_loader(currency='EUR_USD',startDate='12/25/18',startTime='0000',endDate='12/25/18',endTime='0000',all=False):
    dataFile = []
    labels = []
    assemblyStarted = False
    filename = 'data/clean_'+currency+'_data.csv'

    with open(filename) as csvFile:
        csv_reader = csv.reader(csvFile)
        for row in csv_reader:
            if not assemblyStarted:
                if (startDate == row[0] and startTime == row[1]) or all:
                    labels.append(row[0])
                    dataFile.append(np.array([float(row[1]),
                                            float(row[3]),
                                            float(row[4]),
                                            float(row[5]),
                                            float(row[7]),
                                            float(row[8]),
                                            float(row[9]),
                                            float(row[10])]))
                    assemblyStarted = True
            else:
                if endDate == row[0] and endTime == row[1]:
                    break
                else:
                    labels.append(row[0])
                    dataFile.append(np.array([float(row[1]),
                                            float(row[3]),
                                            float(row[4]),
                                            float(row[5]),
                                            float(row[7]),
                                            float(row[8]),
                                            float(row[9]),
                                            float(row[10])]))


    if len(dataFile) < 2:
        print("Error making the data file")

    print(len(dataFile)/1440)
    return np.array(dataFile), np.array(labels)

#
