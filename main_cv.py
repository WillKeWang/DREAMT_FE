# load packages
import pandas as pd
import numpy as np
import random
import shap
from sklearn.model_selection import KFold

from utils import *
from models import *
from datasets import *

import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

np.random.seed(0)

# Prepare the data
# Adjust your path here
quality_df_dir = './results/quality_scores_per_subject.csv'
features_dir = "dataset_sample/E4_aggregate/"
info_dir = "/media/nvme1/sleep/DREAMT_Version2/participant_info.csv"
clean_df, new_features, good_quality_sids = data_preparation(0.2, quality_df_dir, features_dir, info_dir)

# Split data to train, validation, and test set
SW_df, final_features = split_data(clean_df, good_quality_sids, new_features)
print(SW_df.shape)
print(SW_df.Sleep_Stage.value_counts())

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=0)

group_variables = ["AHI_Severity", "Obesity"]
group_variable = get_variable(group_variables, idx=0)
print(len(final_features))
result_dfs = []

# Perform the split
for fold, (trainval_idx, test_idx) in enumerate(kf.split(good_quality_sids)):
    # Now split trainval into training and validation
    # Since trainval_idx corresponds to 64 subjects, we'll use the first 48 for training and the next 16 for validation
    train_sids = [good_quality_sids[idx] for idx in trainval_idx[:54]]
    val_sids = [good_quality_sids[idx] for idx in trainval_idx[54:]]
    test_sids = [good_quality_sids[idx] for idx in test_idx]

    print(f"Fold {fold + 1}")

    X_train, y_train, group_train = train_test_split(SW_df, train_sids, final_features, group_variable)
    X_val, y_val, group_val = train_test_split(SW_df, val_sids, final_features, group_variable)
    X_test, y_test, group_test = train_test_split(SW_df, test_sids, final_features, group_variable)

    # Resample all the data
    X_train_resampled, y_train_resampled, group_train_resampled = resample_data(X_train, y_train, group_train, group_variable)

    # Run LightGBM model
    final_lgb_model = LightGBM_engine(X_train_resampled, y_train_resampled, X_val, y_val)
    # calculate training scores
    prob_ls_train, len_train, true_ls_train = compute_probabilities(
        train_sids, SW_df, final_features, "lgb", final_lgb_model, group_variable)
    lgb_train_results_df = LightGBM_result(final_lgb_model, X_train, y_train, prob_ls_train, true_ls_train)

    # calculate testing scores
    prob_ls_test, len_test, true_ls_test = compute_probabilities(
        test_sids, SW_df, final_features, "lgb", final_lgb_model, group_variable)
    lgb_test_results_df = LightGBM_result(final_lgb_model, X_test, y_test, prob_ls_test, true_ls_test)

    # Add LSTM for post processing
    # create train data
    dataloader_train = LSTM_dataloader(
        prob_ls_train, len_train, true_ls_train, batch_size=32
    )

    # Run LSTM model
    LSTM_model = LSTM_engine(dataloader_train, num_epoch=300, hidden_layer_size=32, learning_rate=0.001) # set your num_epoch

    # test LSMT model
    dataloader_test = LSTM_dataloader(
        prob_ls_test, len_test, true_ls_test, batch_size=1
    )
    lgb_lstm_test_results_df = LSTM_eval(LSTM_model, dataloader_test, true_ls_test, 'LightGBM_LSTM')


    # Run GPBoost model
    final_gpb_model = GPBoost_engine(X_train_resampled, group_train_resampled, y_train_resampled, X_val, y_val, group_val)
    # calculate training scores
    prob_ls_train, len_train, true_ls_train = compute_probabilities(
        train_sids, SW_df, final_features, 'gpb', final_gpb_model, group_variable)
    gpb_train_results_df = GPBoost_result(final_gpb_model, X_train, y_train, group_train, prob_ls_train, true_ls_train)

    # calculate testing scores
    prob_ls_test, len_test, true_ls_test = compute_probabilities(
        test_sids, SW_df, final_features, 'gpb', final_gpb_model, group_variable)
    gpb_test_results_df = GPBoost_result(final_gpb_model, X_test, y_test, group_test, prob_ls_test, true_ls_test)


    # Get LSTM dataset
    dataloader_train = LSTM_dataloader(
        prob_ls_train, len_train, true_ls_train, batch_size=32
    )
    dataloader_test = LSTM_dataloader(
        prob_ls_test, len_test, true_ls_test, batch_size=1
    )

    # Run LSTM model
    LSTM_model = LSTM_engine(dataloader_train, num_epoch=300, hidden_layer_size=32, learning_rate = 0.001) # set your num_epoch
    gpb_lstm_test_results_df = LSTM_eval(LSTM_model, dataloader_test, true_ls_test, 'GPBoost_LSTM')

    # overall result
    result_df = pd.concat([lgb_test_results_df, lgb_lstm_test_results_df, 
                                gpb_test_results_df, gpb_lstm_test_results_df])
    result_dfs.append(result_df)
    result_df.to_csv(f'./results/fold_{fold + 1}_results_AHI.csv', index=False)
