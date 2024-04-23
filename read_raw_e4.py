import os
import time
import math
import datetime
import pandas as pd
import numpy as np
import warnings

# helper functions on timestamp manipulation
def string2obj(datetime_str):
    """convert string of datetime to python datetime object and return time only
    Args:
        datetime_str (str): _description_
    Returns:
        datetime object: _description_
    """
    return datetime.datetime.strptime(
        datetime_str, '%H:%M:%S').time()

def timestamp2obj(timestamp):
    """convert timestamp in seconds to datetime object
    Args:
        timestamp (float): _description_
    Returns:
        _ (float): timestamp in seconds
    Note:
        assuming summer time and -4 time zone
    """
    return datetime.datetime.utcfromtimestamp(timestamp-4*3600).time()


class E4Folder:
    ''' Open a folder containing the data from a E4 recording
        and download or segment the time series data

        Also includes how to match to recorded events 
    '''

    def __init__(self, folder_dir, sleep_dir, report_dir, sid):
        """ Initialize the E4folder object

        Parameters
        ----------
        folder_dir : str
            string of where the folder containing one E4 recording is located
        sleep_dir : str
            string of where the folder containing one hand-labeled sleep stages 
            file is located
        report_dir : str
            string of the directory of a csv file documenting the events/labels
            of this E4 dataset
        sid : str
            string of the participant number for matching its data and label
        
        Returns
        -------
        self : E4Folder obj
            Object saving all relevant information regarding this E4 record

        Example
        -------
        folder_dir = './raw_data/E4_data/{}/Empatica/'.format(sid)
        sleep_dir = './raw_data/sleep_label/{}.csv'.format(sid)
        report_dir = './raw_data/APNEA/{} - report.csv'.format(sid)
        sid_E4_folder = E4Folder(folder_dir, sleep_dir, report_dir, sid)
        final_df, IBI_df = sid_E4_folder.load()
        """

        self.folder_dir = folder_dir
        self.sleep_dir = sleep_dir
        self.report_dir = report_dir
        self.sid = sid
        self.temp_filename = self.folder_dir + self.sid + '/TEMP.csv'
        self.hr_filename = self.folder_dir + self.sid + '/HR.csv'
        self.acc_filename = self.folder_dir + self.sid + '/ACC.csv'
        self.bvp_filename = self.folder_dir + self.sid + '/BVP.csv'
        self.eda_filename = self.folder_dir + self.sid + '/EDA.csv'
        self.ibi_filename = self.folder_dir + self.sid + '/IBI.csv'
        self.final_df = None
        self.IBI_df = None

    def read_event_timestamps(self):
        """ Read the file recording the labels/events 
        Returns
        -------
        labels_dict : dict
            dictionary with key = labels (events) and values = list of timestamps
        labels_timestamp: dict
            dictionary with key = labels (events) and values = start timestamp
        """
        # get sleep stage path
        path = str(self.sleep_dir + self.sid + '.csv')

        # get recorded patient sleep stages
        subject_sleep_stages_df = pd.read_csv(path)

        # Convert time information (str) to TIMESTAMP (datetime)
        subject_sleep_stages_df['TIMESTAMP'] = subject_sleep_stages_df['Time'].apply(string2obj)
        return subject_sleep_stages_df

    def read_epoch_report(self):
        """ Read the file containing sleep label from PSG report
        Returns
        -------
        sleep_stage : pandas dataframe
            dataframe containing sleep stage and its epoch
            columns: epoch, sleep_stage
            
        """
        # read epoch report
        epoch_report = str(self.report_dir + self.sid + " - Epoch Report.txt")
        epoch = []
        stage = []
        # open txt file
        with open(epoch_report, 'r') as file:
            # skip header
            for _ in range(21):
                next(file)
            # read each line
            for line in file:
                line_list = line.strip().split()
                try:
                    if line_list[0] != '#':
                        # find sleep period by stage
                        if line_list[-2] != 'L' and line_list[-2] != 'N/A':
                            # get epoch index
                            epoch.append(int(line_list[0]))
                            # get sleep stage
                            stage.append(line_list[-2])
                except IndexError:
                    continue
        sleep_stage = pd.DataFrame({'epoch':epoch, 'sleep_stage':stage})
        return sleep_stage

    def combine_stage_epoch(self):
        """ Combine sleep stage and epoch
        Returns
        -------
        df : pandas dataframe
            dataframe containing sleep stages and apnea and timestamp
            columns: TIMESTAMP, Sleep_Stage, epoch
            
        """
        # get sleep label from report
        report_label = self.read_epoch_report()
        # get label from report
        raw_label = self.read_event_timestamps()

        df = raw_label.copy()
        if raw_label.shape[0] == report_label.shape[0]:
            df['Sleep_Stage'] = report_label['sleep_stage']
            df['epoch'] = report_label['epoch']
        else:
            start = raw_label[raw_label['Sleep_Stage'].isna()].index[0]
            end = raw_label[raw_label['Sleep_Stage'].isna()].index[-1]
            df['epoch'] = None
            df.Sleep_Stage[:start] = report_label.sleep_stage[:start]
            df.Sleep_Stage[end+1:] = report_label.sleep_stage[start:]

            df.epoch[:start] = report_label.epoch[:start]
            df.epoch[end+1:] = report_label.epoch[start:]

        df.Sleep_Stage = df.Sleep_Stage.fillna('Missing')
        df = df[['TIMESTAMP', 'Sleep_Stage', 'epoch']]
        return df

    def read_temp(self):
        """ Read the file recording the skin temperature time series data
        Returns
        -------
        TEMP_df : pandas dataframe
            dataframe containing skin temperature readings from a E4 recording
            columns: TIMESTAMP, TEMP
        """
        filename = self.temp_filename
        # read skin temperature from E4
        TEMP_df = pd.read_csv(filename,
                              skiprows=2, header=None)
        f = open(filename, 'r')
        # get starting time
        line1 = f.readline()
        start_timestamp = int(float(line1.strip()))
        # get sampling frequency
        line2 = f.readline()
        frequency = int(float(line2.strip()))

        TEMP_df.columns = ["TEMP"]
        # Create timestamps
        N = TEMP_df.shape[0]
        timestamps = np.linspace(0, N-1, N)/frequency + start_timestamp
        TEMP_df["TIMESTAMP"] = timestamps
        TEMP_df = TEMP_df.loc[:, ["TIMESTAMP", "TEMP"]]

        # Convert timestamp to datetime
        TEMP_df["TIMESTAMP"] = TEMP_df["TIMESTAMP"].apply(timestamp2obj)
        return TEMP_df

    def read_acc(self):
        """ Read the file recording the accelerometry time series data
        Returns
        -------
        ACC_df : pandas dataframe
            dataframe containing accelerometry readings from a E4 recording
            columns: TIMESTAMP, ACC_X, ACC_Y, ACC_Z
        """
        filename = self.acc_filename
        ACC_df = pd.read_csv(filename,
                             skiprows=2, header=None)
        f = open(filename, 'r')
        line1 = f.readline()
        line2 = f.readline()
        start_timestamp = int(float(line1.split(',')[0]))
        frequency = int(float(line2.split(',')[0]))
        ACC_df.columns = ["ACC_X", "ACC_Y", "ACC_Z"]
        N = ACC_df.shape[0]
        timestamps = np.linspace(0, N-1, N)/frequency + start_timestamp
        ACC_df["TIMESTAMP"] = timestamps
        ACC_df = ACC_df.loc[:, ["TIMESTAMP", "ACC_X", "ACC_Y", "ACC_Z"]]
        ACC_df["TIMESTAMP"] = ACC_df["TIMESTAMP"].apply(timestamp2obj)
        return ACC_df

    def read_ibi(self):
        """ Read the file recording the Inter-beat-interval event data

        Returns
        -------
        IBI_df : pandas dataframe
            dataframe containing IBI readings from a E4 recording
        start_timestamp : float
            timestamp at the start of the recording
        """
        filename = self.ibi_filename
        IBI_df = pd.read_csv(filename,
                             skiprows=3, header=None)
        IBI_df.columns = ["TIMESTAMP", "IBI"]
        f = open(filename, 'r')
        line1 = f.readline()
        start_timestamp = int(float(line1.split(',')[0]))
        IBI_df.TIMESTAMP = IBI_df.TIMESTAMP  + start_timestamp
        IBI_df["TIMESTAMP"] = IBI_df["TIMESTAMP"].apply(timestamp2obj)
        return IBI_df

    def read_bvp(self):
        """ Read the file recording the blood volume pulse time series data

        Returns
        -------
        BVP_df : pandas dataframe
            dataframe containing BVP from a E4 recording
            columns: TIMESTAMP, BVP
        """
        filename = self.bvp_filename
        BVP_df = pd.read_csv(filename,
                             skiprows=2, header=None)
        f = open(filename, 'r')
        line1 = f.readline()
        line2 = f.readline()
        start_timestamp = int(float(line1.strip()))
        frequency = int(float(line2.strip()))
        BVP_df.columns = ["BVP"]
        N = BVP_df.shape[0]
        timestamps = np.linspace(0, N-1, N)/frequency + start_timestamp
        BVP_df["TIMESTAMP"] = timestamps
        BVP_df = BVP_df.loc[:, ["TIMESTAMP", "BVP"]]
        BVP_df["TIMESTAMP"] = BVP_df["TIMESTAMP"].apply(timestamp2obj)
        return BVP_df

    def read_eda(self):
        """ Read the file recording the electrodermal activity time series data

        Returns
        -------
        EDA_df : pandas dataframe
            dataframe containing EDA readings from a E4 recording
            columns: TIMESTAMP, EDA
        """
        filename = self.eda_filename
        EDA_df = pd.read_csv(filename,
                             skiprows=2, header=None)
        f = open(filename, 'r')
        line1 = f.readline()
        line2 = f.readline()
        start_timestamp = int(float(line1.strip()))
        frequency = int(float(line2.strip()))
        EDA_df.columns = ["EDA"]
        N = EDA_df.shape[0]
        timestamps = np.linspace(0, N-1, N)/frequency + start_timestamp
        EDA_df["TIMESTAMP"] = timestamps
        EDA_df = EDA_df.loc[:, ["TIMESTAMP", "EDA"]]
        EDA_df["TIMESTAMP"] = EDA_df["TIMESTAMP"].apply(timestamp2obj)
        return EDA_df

    def read_hr(self):
        """ Read the file recording the heart rate time series data

        Returns
        -------
        TEMP_df : pandas dataframe
            dataframe containing HR readings from a E4 recording
            columns: TIMESTAMP, HR
        """
        filename = self.hr_filename
        HR_df = pd.read_csv(filename,
                            skiprows=2, header=None)
        f = open(filename, 'r')
        line1 = f.readline()
        line2 = f.readline()
        start_timestamp = int(float(line1.strip()))
        frequency = int(float(line2.strip()))
        HR_df.columns = ["HR"]
        N = HR_df.shape[0]
        timestamps = np.linspace(0, N-1, N)/frequency + start_timestamp
        HR_df["TIMESTAMP"] = timestamps
        HR_df = HR_df.loc[:, ["TIMESTAMP", "HR"]]
        HR_df["TIMESTAMP"] = HR_df["TIMESTAMP"].apply(timestamp2obj)
        return HR_df

    def get_E4_stage_df(self):
        """ Read all the E4 data files and merge them into one dataframe,
            and comebine with the recorded sleep stage

        Returns
        -------
        final_e4_stage : pandas dataframe
            dataframe containing all E4 data
            columns: TIMESTAMP, BVP, ACC_X, ACC_Y, ACC_Z, TEMP, EDA, HR, 
                     IBI, Sleep_Stage
        """
        # get final E4 dataframe
        HR_df = self.read_hr()
        ACC_df = self.read_acc()
        BVP_df = self.read_bvp()
        TEMP_df = self.read_temp()
        BVP_df = self.read_bvp()
        EDA_df = self.read_eda()
        IBI_df = self.read_ibi()

        e4_stage = BVP_df.merge(ACC_df, on='TIMESTAMP', how='left')
        e4_stage = e4_stage.merge(TEMP_df, on='TIMESTAMP', how='left')
        e4_stage = e4_stage.merge(EDA_df, on='TIMESTAMP', how='left')
        e4_stage = e4_stage.merge(HR_df, on='TIMESTAMP', how='left')
        e4_stage = e4_stage.merge(IBI_df, on='TIMESTAMP', how='left')

        # get sleep stage with timestamp and epoch
        df = self.combine_stage_epoch()

        # combine E4 and sleep stage and epoch
        final_e4_stage = e4_stage.merge(df, on='TIMESTAMP', how='left')

        return final_e4_stage

    def combine_apnea_report(self):
        """ Read the file containing apnea event from PSG report, and add
            them to the dataframe containing all the E4 data and sleep stage

        Returns
        -------
        e4_stage : pandas dataframe
            Dataframe containing all E4 data, sleep stage and apnea event
            columns: TIMESTAMP, BVP, ACC_X, ACC_Y, ACC_Z, TEMP, EDA, HR, 
                     IBI, Sleep_Stage, Obstructive_Apnea, Central_Apnea,
                     Hypopnea, Multiple_Events
        """
        apnea_report = str(self.report_dir + self.sid + " - Apnea Hypopnea Report.txt")
        e4_stage = self.get_E4_stage_df()
        e4_stage['Obstructive_Apnea'] = np.nan
        e4_stage['Central_Apnea'] = np.nan
        e4_stage['Hypopnea'] = np.nan
        e4_stage['Multiple_Events'] = np.nan

        with open(apnea_report, 'r') as file:
            # skip header
            for _ in range(15):
                next(file)
            for line in file:
                try:
                    # get starting time
                    start = float(line.strip().split()[1].split('s')[0])
                    # get duration
                    duration = float(line.strip().split()[2].split('s')[0])
                    # calculate the duration of each event
                    time = start + duration
                    # get index of event starting
                    index = int(line.strip().split()[0])

                    # get index
                    df_idx = np.where(e4_stage['epoch'] == index)[0][0]
                    start_idx = int(df_idx + start*64)
                    end_idx = int(start_idx + duration*64)

                    # get apnea label
                    label = line.strip().split()[3]
                    if label == '--':
                        existing_event = e4_stage.loc[start_idx:end_idx, 'Hypopnea']
                        try: 
                            if existing_event.isna().sum() == existing_event.shape[0]:
                                e4_stage.loc[start_idx:end_idx, 'Hypopnea'] = 1
                            # print(existing_event.value_counts().shape)
                        except:
                            print(self.sid)
                            print(existing_event.value_counts())
                    elif label == 'O':
                        existing_event = e4_stage.loc[start_idx:end_idx, 'Obstructive_Apnea']
                        try: 
                            if existing_event.isna().sum() == existing_event.shape[0]:
                                e4_stage.loc[start_idx:end_idx, 'Obstructive_Apnea'] = 1
                            # print(existing_event.value_counts().shape)
                        except:
                            print(self.sid)
                            print(existing_event.value_counts())
                    elif label == 'C':
                        existing_event = e4_stage.loc[start_idx:end_idx, 'Central_Apnea']
                        try: 
                            if existing_event.isna().sum() == existing_event.shape[0]:
                                e4_stage.loc[start_idx:end_idx, 'Central_Apnea'] = 1
                            # print(existing_event.value_counts().shape)
                        except:
                            print(self.sid)
                            print(existing_event.value_counts())
                    elif label == 'M':
                        existing_event = e4_stage.loc[start_idx:end_idx, 'Multiple_Events']
                        try: 
                            if existing_event.isna().sum() == existing_event.shape[0]:
                                e4_stage.loc[start_idx:end_idx, 'Multiple_Events'] = 1
                            # print(existing_event.value_counts().shape)
                        except:
                            print(self.sid)
                            print(existing_event.value_counts())

                except IndexError:
                    continue
                except ValueError:
                    continue

        e4_stage = e4_stage.drop('epoch', axis=1)
        return e4_stage

    def get_final_df(self):
        """ Combine all the data to one dataframe and fill in the sleep stage according
            to time

        Returns
        -------
        final_df : pandas dataframe
            Dataframe containing all E4 data, sleep stage and apnea event
            columns: TIMESTAMP, BVP, ACC_X, ACC_Y, ACC_Z, TEMP, EDA, HR, 
                     IBI, Sleep_Stage, Obstructive_Apnea, Central_Apnea,
                     Hypopnea, Multiple_Events
        """
        final_df = self.combine_apnea_report()
        sleep_stages_df = self.combine_stage_epoch()

        # fill Na with previous value for E4
        final_df.iloc[:, :9] = final_df.iloc[:, :9].fillna(method="ffill")

        # find final wake time
        wake_time = list(sleep_stages_df.TIMESTAMP)[-1]
        wake_index = np.where(final_df['TIMESTAMP'] == wake_time)[0][0]

        # fill NA with previous value for sleep label
        final_df.iloc[:wake_index, 9] = final_df.iloc[:wake_index, 9].fillna(method="ffill")

        # fill Na with Preparation epochs
        final_df.Sleep_Stage = final_df.Sleep_Stage.fillna('P')

        # hr always start late
        length_to_delete = len(np.where(final_df.HR.isna())[0])
        final_df = final_df.loc[(length_to_delete):, :]

        # remove the signals after the last annotated sleep stage:
        final_df = final_df.loc[
            : final_df[
                final_df["Sleep_Stage"].isin(["N1", "N2", "N3", "R", "W"])
            ].last_valid_index(),
            :,
        ]

        final_df.reset_index(drop=True, inplace=True)
        return final_df

    def adjust_time(self):
        """ Deidentify the time information in the fianl dataframe

        Returns
        -------
        final_df : pandas dataframe
            Dataframe containing all E4 data, sleep stage and apnea event,
            time information is deidentified. 
            columns: TIMESTAMP, BVP, ACC_X, ACC_Y, ACC_Z, TEMP, EDA, HR, 
                     IBI, Sleep_Stage, Obstructive_Apnea, Central_Apnea,
                     Hypopnea, Multiple_Events
        """
        final_df = self.get_final_df()
        # change time to timestamp
        timestamp = [0 + i * 1/64 for i in range(len(final_df))]
        final_df['TIMESTAMP'] = timestamp
        final_df.reset_index(drop=True, inplace=True)

        return final_df

    def load(self):
        """ Run all the E4 data preparing functions

        -------
        final_df : pandas dataframe
            Dataframe containing all E4 data, sleep stage and apnea event,
            time information is deidentified. 
            columns: TIMESTAMP, BVP, ACC_X, ACC_Y, ACC_Z, TEMP, EDA, HR, 
                     IBI, Sleep_Stage, Obstructive_Apnea, Central_Apnea,
                     Hypopnea, Multiple_Events
        """
        final_df = self.adjust_time()
        self.final_df = final_df
        return final_df


def main():
    # adjust your folder path here
    folder_dir = './E4_data/'
    sleep_dir = "./sleep_stage/"
    report_dir = "./DREAMT_APNEA/"
    info = pd.read_csv("./participant_info.csv")

    aggre_path = "dataset/E4_aggregate/"

    for sid in info.SID:
        E4 = E4Folder(folder_dir, sleep_dir, report_dir, sid)
        final_df = E4.load()
        path = str(aggre_path + sid + "_whole_df.csv")
        final_df.to_csv(path, index=False)
        print(sid + " read data success")

    pass

if __name__ == "__main__":
    main()
