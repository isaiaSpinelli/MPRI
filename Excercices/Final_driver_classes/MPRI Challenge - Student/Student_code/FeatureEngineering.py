import math
import pandas as pd
import numpy as np

#All the events to consider (we do not use Trainign 2)
all_events = ['Training 1', 'Driving']

def compute_EDA_features(data_df, segmentation_level = 1):
    """
    The method computes/extracts features from raw EDA data
        :param data_df: the dataframe to process
        :param segmentation_level: the number of segments the Driving session will be divided in. A segmentation_level of corresponds to no segmentation. A segmentation_level of 20 means that the driving session will be divided in 20 small segments.
        :return: the features as a panda serie (= vector), when segmented the column names start with the id of the segment (s_{})!
    """
    print("----- Computing EDA features (segmentation={}) -----".format(segmentation_level))
    features_df = pd.DataFrame()
    baseline_df = data_df[data_df.event == 'Training 1']
    driving_df = data_df[data_df.event == 'Driving']

    chunk_size = int(driving_df.shape[0] / segmentation_level)

    segment_count = 1
    for start in range(0, driving_df.shape[0], chunk_size):
        # Divide the driving session in segments when segmentation_lvl > 1
        df_subset = driving_df.iloc[start:start + chunk_size]
        # TODO You should compute/extarct additional features from the EDA raw signal here
        # EDA feature ?

        # MIN
        features_df['s{}_EDA_min_Bl'.format(segment_count)] = [baseline_df["EDA"].min()]
        features_df['s{}_EDA_min_Dr'.format(segment_count)] = [df_subset["EDA"].min()]
        # MIN DIFF
        # min_d = df_subset["EDA"].min()
        # min_b = baseline_df["EDA"].min()
        # features_df['s{}_EDA_min_diff'.format(segment_count)] = [min_d - min_b]

        # MEAN
        #features_df['s{}_EDA_mean_Dr'.format(segment_count)] = [df_subset["EDA"].mean()]
        #features_df['s{}_EDA_mean_Bl'.format(segment_count)] = [baseline_df["EDA"].mean()]

        # MEAN DIFF
        # mean_d = df_subset["EDA"].mean()
        # mean_b = baseline_df["EDA"].mean()
        # features_df['s{}_EDA_mean_diff'.format(segment_count)] = [mean_d - mean_b]

        # STD DIFF
        std_d = df_subset["EDA"].std()
        std_b = baseline_df["EDA"].std()
        features_df['s{}_EDA_std_diff'.format(segment_count)] = [std_d - std_b]

        # MEDIAN DIFF
        median_d = df_subset["EDA"].median()
        median_b = baseline_df["EDA"].median()
        features_df['s{}_EDA_median_diff'.format(segment_count)] = [median_d - median_b]

        segment_count+=1
        if segment_count > segmentation_level:
            break
    print("----- Computing EDA features successfully completed -----")
    return features_df

def compute_ECG_features(data_df, segmentation_level = 1):
    """
        The method computes/extract features from raw ECG data
            :param data_df: the dataframe to process
            :param segmentation_level: the number of segments the Driving session will be divided in. A segmentation_level of corresponds to no segmentation. A segmentation_level of 20 means that the driving session will be divided in 20 small segments.
            :return: the features as a panda serie (= vector)
        """
    print("----- Computing ECG features (segmentation={}) -----".format(segmentation_level))
    hrw = 0.075  # One-sided window size, to Calculate moving average
    fs = 1000  # Sample rate

    # print("----- Calculating rolling mean of ECG signal -----")
    mov_avg = []
    data_df['ECG_rollingmean'] = 0
    mov_avg = data_df['ECG'].rolling(int(hrw * fs)).mean()
    avg_hr = np.mean(data_df['ECG'])
    mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
    data_df['ECG_rollingmean'] = mov_avg

    all_local_events = all_events.copy()
    # Simpler solution is to rename the different parts of the Driving into Driving_s{i}
    baseline_df = data_df[data_df.event == 'Training 1']
    driving_df = data_df[data_df.event == 'Driving'].copy() # need to copy to avoid warning
    chunk_size = int(driving_df.shape[0] / segmentation_level)
    segment_count = 1

    all_local_events.remove("Driving")
    event_col_index = driving_df.columns.get_loc("event")
    for start in range(0, driving_df.shape[0], chunk_size):
        driving_segment = 'Driving_s{}'.format(segment_count)
        driving_df.iloc[start:start + chunk_size, event_col_index] = driving_segment
        segment_count += 1
        all_local_events.append(driving_segment)
        if segment_count > segmentation_level:
            break

    features_df = pd.DataFrame()
    measures_ECG = {}
    fs_list = {}

    # key = subject_id ; df = dataset of each subject

    for event in all_local_events:
        if event == 'Training 1':
            sub_sub_df = baseline_df
        else:
            sub_sub_df = driving_df[driving_df.event == event]
        sampletimer = [x for x in sub_sub_df.time]  # dataset.timer is a ms counter with start of recording at '0'
        fs_list[event] = ((len(sampletimer) / (sampletimer[-1] - sampletimer[0])))  # Divide total length of dataset by last timer entry.

    # Calculate moving average with 0.075s in both directions
    for event in all_local_events:
        # print("Searching for optimal offset of moving average for {}".format(event))
        valid_ma = []
        if event == 'Training 1':
            sub_sub_df = baseline_df
        else:
            sub_sub_df = driving_df[driving_df.event == event]
        measures_ECG[event] = {}

        # Trying different values of moving average offset, from 0.1mV to 0.35mV by step of 0.05mV
        ma_offset = 0.1
        for i in range(0, 6):
            ma_offset = ma_offset + 0.05
            # for ma_perc in ma_perc_list:
            # rolmean = [(ma_offset + x + (((x+ma_offset)/100)*ma_perc)) for x in sub_sub_df.ECG_rollingmean] #Raise moving average with passed ma_perc
            rolmean = [(ma_offset + x) for x in sub_sub_df.ECG_rollingmean]  # Raise moving average with passed ma_perc

            # Detection of beats and adding beat position to peaklist
            window = []
            peaklist = []
            listpos = 0
            for datapoint in sub_sub_df.ECG:
                rollingmean = rolmean[listpos]
                if (datapoint <= rollingmean) and (len(window) <= 1):
                    listpos += 1
                elif (datapoint > rollingmean):
                    window.append(datapoint)
                    listpos += 1
                else:
                    maximum = max(window)
                    beatposition = listpos - len(window) + (window.index(max(window)))
                    peaklist.append(beatposition)
                    window = []
                    listpos += 1
            measures_ECG[event]['peaklist'] = peaklist
            # measures_ECG[subject_id][event]['ybeat'] = [sub_df.ECG[sub_df['time']==(x+sub_df['time'].iloc[0])] for x in peaklist]
            # measures_ECG[subject_id][event]['rolmean'] = rolmean

            # Calculating RR intervals and then adding to RR_list
            RR_list = []
            cnt = 0
            while (cnt < (len(peaklist) - 1)):
                RR_interval = (peaklist[cnt + 1] - peaklist[cnt])
                ms_dist = ((RR_interval / fs_list[event]) * 1000.0)
                RR_list.append(ms_dist)
                cnt += 1
            rr_std_dev = np.std(RR_list)
            bpm = ((len(RR_list) / (len(sub_sub_df.ECG) / fs_list[event])) * 60)
            # print (str(event) + " - BPM : " + str(bpm))

            # adding offset to valid_ma if standard deviation > 1 and 150>bpm>30
            if ((rr_std_dev > 1) and ((bpm > 30) and (bpm < 150))):
                # print("valid rrsd : " + str(rrsd))
                valid_ma.append([rr_std_dev, ma_offset])
        # print(valid_ma)
        # If no valid offset is found, the participant is excluded
        if not valid_ma:
            print("Warning, no valid offset found for current participant")
            # excluded.append(subject_id)
        # Otherwise, we keep the offset that goes with the lowest rrsd
        else:
            measures_ECG[event]['best'] = min(valid_ma, key=lambda t: t[0])[1]  # Save the ma_offset
            # print("Offset and percent of moving average that goes with the lowest rrsd : " + str(measures_ECG[event]['best']))

    # measures_ECG[peaks] contains the time of the ECG peak event
    # measures_RR contains the time of the respiration peak event

    for event in all_local_events:
        # print("Computing moving average with optimal parameters and list of RR intervals for " + str(event))
        rolmean = []
        if event == 'Training 1':
            sub_sub_df = baseline_df
        else:
            sub_sub_df = driving_df[driving_df.event == event]

        ma_offset = measures_ECG[event]['best']
        # ma_perc = measures_ECG[subject_id][event]['best'][1]
        # print(ma_offset)
        # print(ma_perc)
        # rolmean = [(x+((x/100)*ma_perc)) for x in sub_sub_df.ECG_rollingmean] #Raise moving average with passed ma_perc
        # rolmean = [(ma_offset + x + (((x+ma_offset)/100)*ma_perc)) for x in sub_sub_df.ECG_rollingmean] #Raise moving average with passed ma_perc            window = []
        rolmean = [(ma_offset + x) for x in
                   sub_sub_df.ECG_rollingmean]  # Raise moving average with passed ma_perc

        peaklist = []
        listpos = 0
        for datapoint in sub_sub_df.ECG:
            rollingmean = rolmean[listpos]
            if (datapoint <= rollingmean) and (
                    len(window) <= 1):  # Here is the update in (datapoint <= rollingmean)
                listpos += 1
            elif (datapoint > rollingmean):
                window.append(datapoint)
                listpos += 1
            else:
                maximum = max(window)
                beatposition = listpos - len(window) + (window.index(max(window)))
                peaklist.append(beatposition)
                window = []
                listpos += 1

        pos_init = int(sub_sub_df.iloc[0].time * 1000)
        # for x in range(len(rolmean)):
        # print(x)
        # data_df.ECG_rollingmean[data_df['event'] == event] = rolmean
        # data_df.loc[data_df.event == event, 'ECG_rollingmean'] = rolmean  # modified SR -> Data is not used anywhere
        measures_ECG[event]['peaklist'] = [(pos_init + x) for x in peaklist]
        measures_ECG[event]['ybeat'] = [data_df.ECG[pos_init + x] for x in peaklist]
        measures_ECG[event]['rolmean'] = rolmean
        RR_list = []
        cnt = 0
        while (cnt < (len(peaklist) - 1)):
            RR_interval = (peaklist[cnt + 1] - peaklist[cnt])
            ms_dist = ((RR_interval / fs_list[event]) * 1000.0)
            RR_list.append(ms_dist)
            cnt += 1
        measures_ECG[event]['RR_list'] = RR_list
        bpm = ((len(RR_list) / (len(sub_sub_df.ECG) / fs_list[event])) * 60)
        # print(str(event) + " - BPM : " + str(bpm))

        # print("Detecting outliers in ECG for " + str(event))
        RR_list = measures_ECG[event]['RR_list']  # Get measures
        peaklist = measures_ECG[event]['peaklist']
        ybeat = measures_ECG[event]['ybeat']

        # Set thresholds
        upper_threshold = (np.mean(RR_list) + 300)
        lower_threshold = (np.mean(RR_list) - 300)

        # detect outliers
        cnt = 0
        removed_beats = []
        removed_beats_y = []
        RR2 = []
        peaklist2 = []
        ybeat2 = []
        while cnt < len(RR_list):
            if (RR_list[cnt] < upper_threshold) and (RR_list[cnt] > lower_threshold):
                RR2.append(RR_list[cnt])
                peaklist2.append(peaklist[cnt])
                ybeat2.append(ybeat[cnt])
                cnt += 1
            else:
                removed_beats.append(peaklist[cnt])
                removed_beats_y.append(ybeat[cnt])
                cnt += 1
        # Append corrected RR-list to dictionary
        measures_ECG[event]['RR_list_cor'] = RR2
        measures_ECG[event]['peaklist_cor'] = peaklist2
        measures_ECG[event]['ybeat_cor'] = ybeat2

        RR_list = measures_ECG[event]['RR_list_cor']
        RR_diff = []
        RR_sqdiff = []
        cnt = 0
        while (cnt < (len(RR_list) - 1)):
            RR_diff.append(abs(RR_list[cnt] - RR_list[cnt + 1]))
            RR_sqdiff.append(math.pow(RR_list[cnt] - RR_list[cnt + 1], 2))
            cnt += 1
        measures_ECG[event]['RR_diff'] = RR_diff
        measures_ECG[event]['RR_sqdiff'] = RR_sqdiff

        measures_ECG[event]['BPM'] = round(
            (60 * fs_list[event]) / np.mean(measures_ECG[event]['RR_list_cor']), 3)
        measures_ECG[event]['IBI'] = round(np.mean(measures_ECG[event]['RR_list_cor']), 3)
        measures_ECG[event]['SDNN'] = round(np.std(measures_ECG[event]['RR_list_cor']), 3)
        measures_ECG[event]['SDSS'] = round(np.std(measures_ECG[event]['RR_diff']), 3)
        measures_ECG[event]['RMSSD'] = round(np.sqrt(np.mean(measures_ECG[event]['RR_sqdiff'])), 3)
        measures_ECG[event]['NN50'] = round(
            len([x for x in measures_ECG[event]['RR_diff'] if (x > 50)]), 3)
        measures_ECG[event]['PNN50'] = round(
            float(len([x for x in measures_ECG[event]['RR_diff'] if (x > 50)])) / float(
                len(measures_ECG[event]['RR_diff'])), 3)


    # Select the features we want to return (For each feature, we take the value of the feature and its difference with its baseline value)
    selected_features = ['BPM', 'IBI', 'SDNN', 'SDSS', 'RMSSD', 'NN50', 'PNN50']
    driving_events = all_local_events
    driving_events.remove("Training 1")

    for feature in selected_features:
        for i, event in enumerate(driving_events):
            # I = the ID of the current segment
            features_df['s{}_ECG_{}_Dr'.format(i+1, feature)] = [measures_ECG[event][feature]]
            features_df['s{}_ECG_{}_Dr-Bl'.format(i+1, feature)] = [measures_ECG[event][feature] - measures_ECG['Training 1'][feature]]
    print("----- Computing ECG features completed successfully -----")
    return features_df

def get_label(filename):
    subj_info = pd.DataFrame()
    label_str = filename.split('_')[2].split('.')[0]
    if label_str == 'NST':
        subj_info['label'] = [0]
    elif label_str == 'ST':
        subj_info['label'] = [1]
    return subj_info
