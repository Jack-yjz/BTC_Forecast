# coding:utf-8
import numpy as np
import json
import csv
import datetime
import time
import math




def testDate(moreDay):
    d1 = datetime.datetime(2009, 1, 3)
    delta = datetime.timedelta(days=moreDay)
    n_days = d1+delta
    date_timestamp = n_days.strftime('%Y-%m-%d')    
    time_int = int(time.mktime(n_days.timetuple())) 
    timestamp = time_int*1000
    return timestamp,date_timestamp


def test_with_write(data_list):
    with open(r'btc_linearData.csv',"w",newline='') as csvfile:
        writer = csv.writer(csvfile)
        #先写入columns_name
        writer.writerow(["timestamp", "date", "close"])
        for i in range(len(data_list[0])):            
            writer.writerow([data_list[0][i], data_list[1][i], data_list[2][i]])
            #break


def main():
    futureTime = datetime.datetime(2020, 3, 1)
    btc_born_time = datetime.datetime(2009, 1, 3)
    moreDays = (futureTime-btc_born_time).days
    originalData = range(560, moreDays, 1)
    originalData_list = list(originalData)
    timestamp_list = []
    date_timestamp_list = []
    for day in originalData_list:
        timestamp,date_timestamp = testDate(day)
        timestamp_list.append(timestamp)
        date_timestamp_list.append(date_timestamp)
    
    # 生成预测数据
    price_list = []
    for data in originalData_list:
        y = 10**(5.7281125 * math.log(data,10) - 16.62788617732808)
        price_list.append(y)
    data_list = []
    data_list.append(timestamp_list)
    data_list.append(date_timestamp_list)
    data_list.append(price_list)
    return data_list

if __name__ == '__main__':
    data_list = main()
    test_with_write(data_list)




