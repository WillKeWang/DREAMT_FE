""" Code used to calculate quality score of each participant
"""

import os
import numpy as np
import pandas as pd

feature_path = 'dataset_sample/features_df/'
files = os.listdir(feature_path)
files.sort()

sids = []
total_segments = []
num_excludes = []
percentages = []

for file in files:
    sid = file.split('_')[0]
    file_path = str(feature_path + file)
    df = pd.read_csv(file_path)
    segment_len = len(df)
    segment_exclude = np.sum(df.artifact)
    percentage = segment_exclude / segment_len

    sids.append(sid)
    total_segments.append(segment_len)
    num_excludes.append(segment_exclude)
    percentages.append(percentage)

qs = pd.DataFrame({'sid': sids,
                   'total_segments': total_segments,
                   'num_excludes': num_excludes,
                   'percentage_excludes': percentages})


qs.to_csv('./results/quality_scores_per_subject.csv', index=False)
