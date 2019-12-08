#!/usr/bin/env python
"""

This is a custom data pipeline that houses all functions
that will be used for ingesting, transforming and producing
the golden copy for model training, validating and testing.
"""
##################### YOUR LIBRARIES HERE ####################
import json
import numpy as np
import pandas as pd
import os
import zipfile
from zipfile import ZipFile
from datetime import datetime, date
import time
import requests

################### YOUR FUNCTIONS HERE ######################
# Helper functions for feature engineering:

def decode_stacked_json(stacked_json_string, pos=0, decoder=json.JSONDecoder()):
    """Yield multiple JSON objects and restart parsing from the previous position
    Input must be of the form: {'key1': value1, 'key2': value2}{'key3': value3, 'key4': value4"""
    while True:
        try:
            json_object, pos = decoder.raw_decode(stacked_json_string, pos)
        except json.JSONDecodeError:
            break
        yield json_object
       
        
def after_wake_exercise(class_5min,bedtime_end):
    """returns the number of minutes with medium to high MET scores 
    within the first 3 hours of wake-time as a proxy for whether or not exercise 
    occurs after waking up"""
    if isinstance(class_5min,float) or isinstance(bedtime_end, float):
        return np.nan
    else:
        #convert the str integer into a list of integers
        class_5min_list = list(map(int, class_5min))
        #--take the timestamp from the datetime string, extracts hh:mm data, and converts to a number--#
        wake_hr_min = int(''.join(bedtime_end.split('T')[1][0:5].split(':')))
        #---calculate minutes lapsed since 4am and wake up time---#
        #---rescale minutes into 5 minute intervals to find the number of elements at which to offset class_5min--#
        offset = int(((wake_hr_min - 400)/100)*(60/5))

        #subset observations between wake up and 3 hrs post wake up (24*5=120 min)
        morning_obs = class_5min_list[offset:offset+36]

        #tally total minutes spent in medium to high intensity exercise (3-4)
        total_min = sum([1 for obs in morning_obs if obs >=3])*5
        return total_min


def before_sleep_exercise(class_5min,bedtime_start):
    """returns the number of minutes with medium to high MET scores 
    within the last 3 hours of wake-time as a proxy for whether or not exercise 
    occurs in the evening close to bedtime"""
    if isinstance(class_5min,float) or isinstance(bedtime_start, float):
        return np.nan
    else:
        #convert the str integer into a list of integers
        class_5min_list = list(map(int, class_5min))
        #--take the timestamp from the datetime string, extracts hh:mm data, and converts to a number--#
        sleep_hr_min = int(''.join(bedtime_start.split('T')[1][0:5].split(':')))
        #---calculate minutes lapsed since 4am and wake up time---#
        #---rescale minutes into 5 minute intervals to find the number of elements at which to offset class_5min--#
        offset = int(((sleep_hr_min - 400)/100)*(60/5))

        #subset observations 3 hours before sleep time
        evening_obs = class_5min_list[offset-36:offset]

        #tally total minutes spent in medium to high intensity exercise (3-4)
        total_min = sum([1 for obs in evening_obs if obs >=3])*5
        return total_min


def noon_exercise(class_5min):
    """returns the number of minutes with medium to high MET scores 
    between noon and two local time"""
    if isinstance(class_5min,float):
        return np.nan
    else:
        #convert the str integer into a list of integers
        class_5min_list = list(map(int, class_5min))

        #---calculate minutes lapsed since 4am and 12 (8*12 (5min intervasl in an hr)---#

        offset = 96

        #subset observations between noon and 2 pm
        noon_obs = class_5min_list[offset:offset+24]

        #tally total minutes spent in medium to high intensity exercise (3-4)
        total_min = sum([1 for obs in noon_obs if obs >=3])*5
        return total_min
 
    
def evening_exercise(class_5min):
    """returns the number of minutes with medium to high MET scores 
    between noon and two local time"""
    if isinstance(class_5min, float):
        return np.nan
    else:
        #convert the str integer into a list of integers
        class_5min_list = list(map(int, class_5min))

        #---calculate minutes lapsed since 4am and 5pm (13*12) --there are 12 5min intervasl in an hr---#

        offset = 156

        #subset observations between 5pm and 7 pm post wake up
        eve_obs = class_5min_list[offset:offset+24]

        #tally total minutes spent in medium to high intensity exercise (3-4)
        total_min = sum([1 for obs in eve_obs if obs >=3])*5
        return total_min


def after_midnight(timestamp_str):
    """indicator for whether or not bedtime started after midnight (in early am hours)"""
    try:
        day, hour = timestamp_str.split('T')
        hr = int(hour[0:2]) #extract hr from timestamp
        if 0 <= hr < 6:
            return 1
        else:
            return 0
    except:
        return np.nan

def age_bin(e):
    """Bins: 20s, 30s, 40s, and 50s +"""
    if  20 <= e < 30:
        return "20s"
    elif 30 <= e < 40:
        return "30s"
    elif 40 <= e < 50:
        return "40s"
    elif e > 50:
        return "50s plus"
    else:
        return np.nan
   
    
def height_bin(e):
    """Bins: 0, 150, 160, 170, 180, 190. In centimeters """
    if  0 < e < 150:
        return "less than 150 cm"
    elif 150 <= e <= 160:
        return "150s"
    elif 160 <= e <= 170:
        return "160s"
    elif 170 <= e <= 180:
        return "170s"
    elif 180 <= e <= 190:
        return "180s"
    elif e > 190:
        return "greater than 190 cm"
    else:
        return np.nan
    
    
def weight_bin(e):
    """Bins: 0, 65, 80, 95. In kilograms"""
    if  0 < e <=65:
        return "less than 65 kg"
    elif 65 <= e <= 80:
        return "65 to 80 kg"
    elif 80 <= e <= 95:
        return "80 to 95 kg"
    elif e > 95:
        return "more than 95 kg"
    else:
        return np.nan


def period_over_period(dataframe,usrid,date_col, metric_cols,days_offset):
    """for each datetime in the dataframe's date column, computes the metric column's
    difference from the period days prior:
    returns WoW differences with the corresponding userid and datetime as a tuple"""
    #--first recorded observation date + days_offset
    init_date = min(dataframe[date_col]) + pd.DateOffset(days_offset)
    pop_list = []
    for dt in dataframe[date_col]:
        if dt < init_date: # won't have 7 days prior for the first 7 observations so flag as np.nan
            pop_list.append([usrid,dt]+[np.nan for i in np.arange(len(metric_cols))])
        else:
            try: #---need to account for missing dates
                pop_diff_list = []
                for metric in metric_cols:
                    #grab the value from the prior 7(or n) days -- need to index on 0 since values returns array of array
                    prd_prior = dataframe[dataframe[date_col] == dt + pd.DateOffset(-days_offset)][metric].values[0]
                    #grab the current value
                    prd_current = dataframe[dataframe[date_col] == dt][metric].values[0]
                    #take the diff - list of differences
                    try:
                        pop_diff = (prd_current/prd_prior)-1
                    except ZeroDivisionError: #1 if prior period was 0 (basically capping at 100% increase)
                        pop_diff = 1
                   
                    pop_diff_list.append(pop_diff)
                    
                pop_list.append([usrid,dt]+[i for i in pop_diff_list])
            except:
                #---if missing dates then use nan values
                pop_list.append([usrid,dt]+[np.nan for i in np.arange(len(metric_cols))])

    return pop_list

def create_all_features(zipfl,fname,full_path,log_file):
    # define intermediate data frames:
    users = pd.DataFrame()
    sleep = pd.DataFrame()
    readiness = pd.DataFrame()
    activity = pd.DataFrame()
    experiments = pd.DataFrame()
    

    start = time.time()
    raw_data = str(zipfile.ZipFile(full_path + zipfl).read(fname), encoding='utf-8')
    log_file.write(time.asctime() + ' - raw data file loaded. Time: ' + f'{(time.time() - start):.1f}' + ' secs.\n')

    start = time.time()
    log_file.write(time.asctime() + ' - creating feature space\n')

    active_user_counter = 0

    for user_json in decode_stacked_json(raw_data):
        # in the stacked json file, each json object corresponds to all data of one user. So here I will perform all 
        # feature transformation that are user-dependent before I append the whole user data to a final data frame.
        user_user = pd.DataFrame.from_records([user_json['userInfo']['userInfo']])

        try: # user signup date may be useful to define if a user model is relevant or not:
            user_user['signupDate'] = pd.to_datetime(user_json['signupDate'])
        except KeyError:
            user_user['signupDate'] = np.nan

        # one hot encode gender data:
        try:
            user_user['gender'] = np.where(user_user['gender'] == 'male', True, False)
        except KeyError:
            user_user['gender'] = np.nan

        user_user.rename(columns={'gender':'is_male'}, inplace=True)

        try:
            user_user['age'] = user_user['age']
        except KeyError:
            user_user['age'] = np.nan

        try:
            user_user['height'] = user_user['height']
        except KeyError:
            user_user['height'] = np.nan

        try:
            user_user['weight'] = user_user['weight']
        except KeyError:
            user_user['weight'] = np.nan

        # send to a final users data frame:
        users = users.append(user_user)

        # skip users without sleep records (strictly experiment users, or users of other hardware)
        if len(user_json['sleep']) == 0: continue
        active_user_counter += 1

        # unpack data for each user in a specific data frame
        user_sleep = pd.DataFrame.from_records([i for i in user_json['sleep']])
        user_readiness = pd.DataFrame.from_records([i for i in user_json['readiness']])
        user_activity = pd.DataFrame.from_records([i for i in user_json['activity']])

        # each DF has a `score` feature. Renaming to avoid later confusion:
        user_sleep.rename(columns = {'score':'sleep_score'}, inplace=True)
        user_activity.rename(columns = {'score':'activity_score'}, inplace=True)
        user_readiness.rename(columns = {'score':'readiness_score'}, inplace=True)

        #### ---------- SLEEP FEATURES ----------- ####
        # sleep features normalizing and transformations:
        user_sleep['awake_norm'] = user_sleep['awake']/user_sleep['duration']
        user_sleep['deep_norm'] = user_sleep['deep']/user_sleep['duration']
        user_sleep['light_norm'] = user_sleep['light']/user_sleep['duration']
        user_sleep['onset_latency_norm'] = user_sleep['onset_latency']/user_sleep['duration']
        user_sleep['rem_norm'] = user_sleep['rem']/user_sleep['duration']
        user_sleep['restless_norm'] = user_sleep['light']/100

        # bins for sleep score to Shiraz's models:
        bins = [0, 75, 85, 100]
        names = ['fair', 'good', 'greate']
        user_sleep['good_sleep'] = pd.cut(user_sleep['sleep_score'], bins=bins, labels=names)

        # creating new sleep features:
        user_sleep['summary_date'] = pd.to_datetime(user_sleep['summary_date'])
        user_sleep['user_id'] = user_user['user_id'][0]
        user_sleep['user_date'] = user_sleep['user_id'] + '|' + user_sleep['summary_date'].astype('str')

        user_sleep.set_index('summary_date', inplace = True, drop=False)

        # build D - 1 and D - 2 scores:
        user_sleep['sleep_score_D-1'] = user_sleep['sleep_score'].shift()[user_sleep.index.shift(1,freq='1D')]
        user_sleep['sleep_score_D-2'] = user_sleep['sleep_score'].shift()[user_sleep.index.shift(1,freq='2D')]
        user_sleep['deep_D-1'] = user_sleep['deep_norm'].shift()[user_sleep.index.shift(1,freq='1D')]
        user_sleep['deep_D-2'] = user_sleep['deep_norm'].shift()[user_sleep.index.shift(1,freq='2D')]
        user_sleep['rem_D-1'] = user_sleep['rem_norm'].shift()[user_sleep.index.shift(1,freq='1D')]
        user_sleep['rem_D-2'] = user_sleep['rem_norm'].shift()[user_sleep.index.shift(1,freq='2D')]

        # build 7, 14, and 21 rolling average scores:
        user_sleep['rol_sleep_score_7d'] = pd.DataFrame.rolling(user_sleep['sleep_score'].shift(1, freq='1D'),
                                                          window=7, min_periods=3).mean()
        # shifted 7 day score rolled feature. This is target feature to predict your average 7 days ahead
        user_sleep['avg_sleep_score_next_week'] = user_sleep['rol_sleep_score_7d'].shift(-1,freq='7D')

        user_sleep['rol_sleep_score_14d'] = pd.DataFrame.rolling(user_sleep['sleep_score'].shift(1, freq='1D'),
                                                          window=14, min_periods=10).mean()
        user_sleep['rol_sleep_score_21d'] = pd.DataFrame.rolling(user_sleep['sleep_score'].shift(1, freq='1D'),
                                                          window=21, min_periods=17).mean()

        # build other 7-day rolling features for peter's models later:
        user_sleep['rol_bedtime_end_delta_7d'] = pd.DataFrame.rolling(user_sleep['bedtime_end_delta'].shift(1, freq='1D'),
                                                                window=7, min_periods=3).mean()
        user_sleep['rol_bedtime_start_delta_7d'] = pd.DataFrame.rolling(user_sleep['bedtime_start_delta'].shift(1, freq='1D'),
                                                                        window=7, min_periods=3).mean()
        user_sleep['rol_onset_latency_7d'] = pd.DataFrame.rolling(user_sleep['onset_latency_norm'].shift(1, freq='1D'),
                                                                  window=7, min_periods=3).mean()
        user_sleep['rol_duration_7d'] = pd.DataFrame.rolling(user_sleep['duration'].shift(1, freq='1D'),
                                                             window=7, min_periods=3).mean()

        # build is_traveling
        user_sleep['is_traveling'] = np.where(user_sleep['timezone'] == 
                                                        user_sleep['timezone'].value_counts().idxmax(), False, True)

        # build rol_bedtime_start_21d
        user_sleep['rol_bedtime_start_21d'] = pd.DataFrame.rolling(user_sleep['bedtime_start_delta'].shift(1, freq='1D'),
                                                                     window=21, min_periods=10).mean()

        # build avg_bedtime_start_delta, create dummy variables for deviation (-3, -2, -1, 1, 2, 3)
        user_sleep['rol_bedtime_start_std_21d'] = pd.DataFrame.rolling(user_sleep['bedtime_start_delta'].shift(1, freq='1D'),
                                                                     window=21, min_periods=10).std()

        user_sleep['bedtime_start_dev'] = (user_sleep['bedtime_start_delta'] - user_sleep['rol_bedtime_start_21d'])/\
                                          user_sleep['rol_bedtime_start_std_21d']

        user_sleep['bedtime_start_dev'] = np.where(user_sleep['bedtime_start_dev'] >= 0, 
                                                            np.ceil(user_sleep['bedtime_start_dev']),
                                                            np.floor(user_sleep['bedtime_start_dev']))
        # cap standard deviations to -3 or +3:
        user_sleep['bedtime_start_dev'] = np.where(user_sleep['bedtime_start_dev'] <= -3,-3, user_sleep['bedtime_start_dev'])

        user_sleep['bedtime_start_dev'] = np.where(user_sleep['bedtime_start_dev'] >= 3,3,
                                                            user_sleep['bedtime_start_dev'])

        user_sleep = pd.get_dummies(user_sleep, columns=['bedtime_start_dev'])

        user_sleep.rename(columns = { 'bedtime_start_dev_-3.0':'bedtime_start_dev-3', 
                                      'bedtime_start_dev_-2.0':'bedtime_start_dev-2', 
                                      'bedtime_start_dev_-1.0': 'bedtime_start_dev-1',
                                      'bedtime_start_dev_1.0': 'bedtime_start_dev+1',
                                      'bedtime_start_dev_2.0': 'bedtime_start_dev+2',
                                      'bedtime_start_dev_3.0': 'bedtime_start_dev+3'},
                                      inplace = True)

        user_sleep.drop(labels = ['rol_bedtime_start_21d','rol_bedtime_start_std_21d'], axis = 1, inplace = True)

        # build avg_bedtime_end_delta, create dummy variables for deviation. (-3, -2, -1, 1, 2, 3)
        user_sleep['rol_bedtime_end_21d'] = pd.DataFrame.rolling(user_sleep['bedtime_end_delta'].shift(1, freq='1D'),
                                                                     window=21, min_periods=10).mean()

        user_sleep['rol_bedtime_end_std_21d'] = pd.DataFrame.rolling(user_sleep['bedtime_end_delta'].shift(1, freq='1D'),
                                                                     window=21, min_periods=10).std()

        user_sleep['bedtime_end_dev'] = (user_sleep['bedtime_end_delta'] - user_sleep['rol_bedtime_end_21d'])/\
                                        user_sleep['rol_bedtime_end_std_21d']

        user_sleep['bedtime_end_dev'] = np.where(user_sleep['bedtime_end_dev'] >= 0, 
                                                 np.ceil(user_sleep['bedtime_end_dev']),
                                                 np.floor(user_sleep['bedtime_end_dev']))
        # cap standard deviations to -3 or +3:
        user_sleep['bedtime_end_dev'] = np.where(user_sleep['bedtime_end_dev'] <= -3,-3,
                                                 user_sleep['bedtime_end_dev'])

        user_sleep['bedtime_end_dev'] = np.where(user_sleep['bedtime_end_dev'] >= 3,3,
                                                 user_sleep['bedtime_end_dev'])

        user_sleep = pd.get_dummies(user_sleep, columns=['bedtime_end_dev'])

        user_sleep.rename(columns = {'bedtime_end_dev_-3.0':'bedtime_end_dev-3', 
                                     'bedtime_end_dev_-2.0':'bedtime_end_dev-2', 
                                     'bedtime_end_dev_-1.0': 'bedtime_end_dev-1',
                                     'bedtime_end_dev_1.0': 'bedtime_end_dev+1',
                                     'bedtime_end_dev_2.0': 'bedtime_end_dev+2',
                                     'bedtime_end_dev_3.0': 'bedtime_end_dev+3'},
                                     inplace = True)

        user_sleep.drop(labels = ['rol_bedtime_end_21d','rol_bedtime_end_std_21d'], axis = 1, inplace = True)

        # build avg_duration, create dummy variables for deviation (-3, -2, -1, 1, 2, 3)
        user_sleep['rol_duration_21d'] = pd.DataFrame.rolling(user_sleep['duration'].shift(1, freq='1D'),
                                                                     window=21, min_periods=10).mean()

        user_sleep['rol_duration_std_21d'] = pd.DataFrame.rolling(user_sleep['duration'].shift(1, freq='1D'),
                                                                     window=21, min_periods=10).mean()

        user_sleep['duration_dev'] = (user_sleep['duration'] - user_sleep['rol_duration_21d'])/\
                                     user_sleep['rol_duration_std_21d']

        user_sleep['duration_dev'] = np.where(user_sleep['duration_dev'] >= 0, 
                                              np.ceil(user_sleep['duration_dev']),
                                              np.floor(user_sleep['duration_dev']))

        # cap standard deviations to -3 or +3:
        user_sleep['duration_dev'] = np.where(user_sleep['duration_dev'] <= -3,-3, user_sleep['duration_dev'])

        user_sleep['duration_dev'] = np.where(user_sleep['duration_dev'] >= 3,3, user_sleep['duration_dev'])

        user_sleep = pd.get_dummies(user_sleep, columns=['duration_dev'])

        user_sleep.rename(columns = {'duration_dev_-3.0':'duration_dev-3', 
                                  'duration_dev_-2.0':'duration_dev-2', 
                                  'duration_dev_-1.0':'duration_dev-1',
                                  'duration_dev_1.0': 'duration_dev+1',
                                  'duration_dev_2.0': 'duration_dev+2',
                                  'duration_dev_3.0': 'duration_dev+3'},
                                  inplace = True)

        user_sleep.drop(labels = ['rol_duration_21d','rol_duration_std_21d'], axis = 1, inplace = True)

        # build dummy to day of the week.
        user_sleep['weekday'] = user_sleep['summary_date'].dt.weekday

        # build is_workday
        user_sleep['is_workday'] = np.where(user_sleep['weekday'] < 5, True, False)

        # one-hot encode weekdays
        user_sleep = pd.get_dummies(user_sleep, columns=['weekday'])

        user_sleep.rename(columns = {'weekday_0':'weekday_mon', 
                                  'weekday_1':'weekday_tue', 
                                  'weekday_2': 'weekday_wed',
                                  'weekday_3': 'weekday_thu',
                                  'weekday_4': 'weekday_fri',
                                  'weekday_5': 'weekday_sat',
                                  'weekday_6': 'weekday_sun'}, 
                                  inplace = True)

        sleep_vars_to_roll = ['onset_latency','duration','breath_average','is_traveling','score_disturbances']

        for feature in sleep_vars_to_roll :
            user_sleep['rol_' + feature + '_7d'] = pd.DataFrame.rolling(user_sleep[feature].shift(1, freq='1D'),
                                                          window=7, min_periods=3).mean()

        user_sleep.set_index('user_date', inplace = True)

        sleep = sleep.append(user_sleep)


        #### ---------- ACTIVITY FEATURES ----------- ####
        # Creating the new activity related features. Because several activity features depends on user info and/or 
        # sleep features, I will add the features needed here:
        user_activity['user_date'] = user_user['user_id'][0] + '|' + user_activity['summary_date'].astype('str')

        user_activity['age'] = user_user['age'][0]
        user_activity['height'] = user_user['height'][0]
        user_activity['weight'] = user_user['weight'][0]
        user_activity['is_male'] = user_user['is_male'][0]

        user_activity['summary_date'] = pd.to_datetime(user_activity['summary_date'])   
        user_activity = user_activity.merge(user_sleep[['bedtime_start', 'bedtime_end', 'summary_date']], 
                                            on='summary_date', how='left')

        user_activity.set_index('summary_date', inplace = True, drop=False)

        user_activity['met_min_medium_plus'] = user_activity['met_min_medium'] + \
                                                        user_activity['met_min_high']

        user_activity['age_bin'] = user_activity['age'].apply(age_bin)
        user_activity['height_bin'] = user_activity['height'].apply(height_bin)
        user_activity['weight_bin'] = user_activity['weight'].apply(weight_bin)
        user_activity['sleep_afterMidnight'] = user_activity['bedtime_start'].apply(after_midnight)
        user_activity['afterwake_exercise_min'] = [after_wake_exercise(c,b) for c,b in \
                                                   zip(user_activity['class_5min'],
                                                       user_activity['bedtime_end'])]
        user_activity['beforesleep_exercise_min'] = [before_sleep_exercise(c,b) for c,b in \
                                                     zip(user_activity['class_5min'],
                                                         user_activity['bedtime_start'])]
        user_activity['noon_exercise_min'] = user_activity['class_5min'].apply(noon_exercise)
        user_activity['eve_exercise_min'] = user_activity['class_5min'].apply(evening_exercise)

        # create activity and sleep variables that will be rolled within a 7 day window
        activity_vars_to_roll = ['cal_total', 'high', 'medium','steps','inactive', 
                                 'non_wear', 'activity_score','met_min_medium','sleep_afterMidnight',
                                 'met_min_high', 'met_min_medium_plus','score_move_every_hour',
                                 'score_stay_active', 'beforesleep_exercise_min',
                                 'afterwake_exercise_min','noon_exercise_min', 'eve_exercise_min']

        for feature in activity_vars_to_roll :
            user_activity['rol_' + feature + '_7d'] = pd.DataFrame.rolling(user_activity[feature].shift(1, freq='1D'),
                                                          window=7, min_periods=3).mean()

        user_activity.set_index('user_date', inplace = True)
        activity = activity.append(user_activity)


        #### ---------- READINESS FEATURES ----------- ####
        # As we are not doing any transformation in the readiness data frame, append directly to the final one:
        user_readiness['user_date'] = user_user['user_id'][0] + '|' + user_readiness['summary_date'].astype('str')
        user_readiness.set_index('user_date', inplace=True)
        readiness = readiness.append(user_readiness)


        #### ---------- EXPERIMENTS FEATURES ----------- ####
        # Sorting the experiment data for the first iteration:
        if len(user_json['previousExperiments']) != 0:
        # for now, only completed experiments are taken. Each feature contains a list of inputs. The unique user ID is 
        # added for later incorporation of this results with the data set.
        # NOTE: we don't have a summary_date here. To get that, unpack the list inside of features 'CheckIns' and get
        # key 'date'.
            user_experiments = pd.DataFrame.from_records([i for i in user_json['previousExperiments']])
            user_experiments['user_id'] = user_user['user_id'][0]

            experiments = experiments.append(user_experiments)

    # log the time and main descriptive statistics of the final datasets for data governance purposes
    log_file.write(time.asctime() + ' - Feature space created. Time: ' + f'{(time.time() - start):.1f}' + ' secs.\n')
    log_file.write(time.asctime() + '     - Total active users: ' + str(active_user_counter) + '\n')
    log_file.write(time.asctime() + '     - ACTIVITY data frame shape: ' + str(activity.shape) + '\n')
    log_file.write(time.asctime() + '     - SLEEP data frame shape: ' + str(sleep.shape) + '\n')
    log_file.write(time.asctime() + '     - READINESS data frame shape: ' + str(readiness.shape) + '\n')
    log_file.write(time.asctime() + '     - EXPERIMENTS data frame shape: ' + str(experiments.shape) + '\n')

    return users,sleep,readiness,activity,experiments



































##################### (END) CODE HERE ####################