# load packages
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score, cohen_kappa_score

import lightgbm as lgb
import gpboost as gpb
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from imblearn.over_sampling import SMOTE

from torch.utils.data import DataLoader

from utils import *

import warnings

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

np.random.seed(1)

def load_data_to_df(
    threshold, quality_df_dir, info_df, features_dir, nan_feature_names, label_names, circadian_features
):
    """
    Loads and processes feature data from CSV files for subjects meeting a 
    quality score threshold, applying several preprocessing steps including 
    rolling standard deviations, Gaussian filtering, and derivative calculation. 
    The function also classifies subjects based on Apnea-Hypopnea Index (AHI) 
    and Body Mass Index (BMI) into predefined categories.

    Parameters:
    ----------
    threshold : float
        The threshold for the percentage of data excluded based on quality scores. 
        Subjects with quality scores below this threshold are considered for analysis.
    quality_df_dir : str
        A path to the file summarizing the percentage of artifacts of each subject's 
        data calculated from features dataframe
    info_df : pandas.DataFrame
        A DataFrame containing demographic and clinical information for the subjects, 
        indexed by SID.
    features_dir : str
        A path to the folder containing all the features
    nan_feature_names : list of str
        Names of features that should be considered as NaN and excluded from the analysis.
    label_names : list of str
        Names of columns in the data that are considered as labels and should not be 
        treated as features.
    circadian_features : list of str
        Names of features related to circadian rhythms, treated separately from other 
        physiological features.

    Returns:
    -------
    all_subjects_fe_df : pandas dataFrame
        A DataFrame containing the processed features for all subjects meeting the 
        quality threshold.
    good_quality_sids : list of str
        A list of subject IDs that met the quality score threshold and were included 
        in the analysis.
    """
    # load quality scores
    quality_df = pd.read_csv(quality_df_dir)
    good_quality_sids = quality_df.loc[
        quality_df.percentage_excludes < float(threshold), "sid"
    ].to_list()

    # load demographic info
    info_df.index = info_df.SID

    # Read example from one subject for further processing
    path = str(features_dir) + str(info_df.SID[0]) + '_domain_features_df.csv'
    example_df = pd.read_csv(path)

    # select features
    feature_names = [
        f
        for f in example_df.columns.tolist()
        if f not in nan_feature_names + label_names + ["sid"]
    ]

    # select physiological features
    physiological_features = [f for f in feature_names if f not in circadian_features]

    # create dataframe for all the subjects' features
    example_df = rolling_stds(example_df, physiological_features, window_size=10)
    example_df = gaussian_filtering(
        example_df, physiological_features, kernel_size=20, std_dev=100
    )
    example_df = add_derivatives(example_df, physiological_features)

    all_subjects_fe_df = pd.DataFrame(columns=example_df.columns)
    for sid in good_quality_sids:
        path = str(features_dir) + sid + '_domain_features_df.csv'
        sid_df = pd.read_csv(path)
        sid_df = rolling_stds(sid_df, physiological_features, window_size=10)
        sid_df = gaussian_filtering(
            sid_df, physiological_features, kernel_size=20, std_dev=100
        )
        sid_df = add_derivatives(sid_df, physiological_features)

        # add apnea target
        subject_AHI = int(info_df.loc[sid, "AHI"])
        if subject_AHI < 5:
            sid_df["AHI_Severity"] = 0
        elif 5 <= subject_AHI < 15:
            sid_df["AHI_Severity"] = 1
        elif 15 <= subject_AHI < 30:
            sid_df["AHI_Severity"] = 2
        else:
            sid_df["AHI_Severity"] = 3

        # add BMI target
        subject_BMI = info_df.loc[sid, "BMI"]
        if subject_BMI >= 35:
            sid_df["Obesity"] = 1
        else:
            sid_df["Obesity"] = 0

        sid_df = sid_df.loc[:sid_df[sid_df['Sleep_Stage'].isin(["N1", "N2", "N3", "R", "W"])].last_valid_index(), :]

        all_subjects_fe_df = pd.concat([all_subjects_fe_df, sid_df], ignore_index=True)
    return all_subjects_fe_df, good_quality_sids


def clean_features(all_subjects_fe_df, info_df, nan_feature_names, label_names):
    """
    Cleans the feature dataframe by updating feature names, mapping sleep stages,
    replacing infinite values with NaN, deleting features with excessive missing values,
    and merging additional demographic information. It prepares the data for further 
    analysis by filtering out unnecessary columns and rows with missing values, and 
    returns a cleaned dataframe along with a list of the names of the features that 
    were retained.

    Parameters:
    ----------
    all_subjects_fe_df : pandas dataFrame
        A DataFrame containing the processed features for all subjects meeting the 
        quality threshold.
    info_df : pandas.DataFrame
        A DataFrame containing demographic and clinical information for the subjects, 
        indexed by SID.
    nan_feature_names : list of str
        Names of features that should be considered as NaN and excluded from the analysis.
    label_names : list of str
        Names of columns in the data that are considered as labels and should not be 
        treated as features.

    Returns:
    -------
    clean_df : pandas DataFrame
        The cleaned dataframe after applying all preprocessing steps.
    new_features : list
        A list of the names of the features that were retained in the cleaned DataFrame.
    """
    # update features
    updated_feature_names = [
        f
        for f in all_subjects_fe_df.columns.tolist()
        if f not in nan_feature_names + label_names + ["sid"]
    ]

    # get feature dataframe
    df = all_subjects_fe_df.loc[
        :,
        updated_feature_names + label_names + ["sid"],
    ]

    df.Sleep_Stage = df.Sleep_Stage.map(
        {
            "N1": "N",
            "N2": "N",
            "W": "W",
            "N3": "N",
            "P": "P",
            "R": "R",
            "Missing": "Missing",
        }
    )
    # replace inf
    df = df.replace([np.inf, -np.inf], np.nan)

    # delete features if contains too many nan values
    na_count_df = df.isna().sum()
    features_to_delete = na_count_df[na_count_df > 2000].index.to_list()
    cleaned_feature_names = [
        f for f in updated_feature_names if f not in features_to_delete
    ]

    # select feature columns
    df = df.loc[:, cleaned_feature_names + label_names + ["sid"]]
    # drop columns with nan
    df = df.dropna(how="any", axis=0)
    # add BMI information
    df = pd.merge(df, info_df.loc[:, ["BMI"]], left_on="sid", right_index=True)

    map_stage_to_num = {"P": 1, "N": 0, "R": 0, "W": 1, "Missing": np.nan}
    df["Sleep_Stage"] = df["Sleep_Stage"].map(map_stage_to_num)
    clean_df = df.dropna()

    new_features = clean_df.columns.to_list()
    new_features.remove("sid")
    new_features.remove("Sleep_Stage")
    new_features.remove("Central_Apnea")
    new_features.remove("Obstructive_Apnea")
    new_features.remove("Multiple_Events")
    new_features.remove("Hypopnea")
    new_features.remove("AHI_Severity")
    new_features.remove("Obesity")
    new_features.remove("BMI")
    new_features.remove("circadian_decay")
    new_features.remove("circadian_linear")
    new_features.remove("circadian_cosine")
    new_features.remove("timestamp_start")

    return clean_df, new_features


def data_preparation(threshold, quality_df_dir, features_dir, info_dir):
    """
    Prepare the data for modeling by using data preparation functions

    Parameters:
    ----------
    threshold : float
        The threshold for the percentage of data excluded based on quality scores. 
        Subjects with quality scores below this threshold are considered for analysis.
    
    Returns:
    -------
    clean_df : pandas DataFrame
        The cleaned dataframe after applying all preprocessing steps.
    new_features : list
        A list of the names of the features that were retained in the cleaned DataFrame.
    good_quality_sids : list of str
        A list of subject IDs that met the quality score threshold and were included 
        in the analysis.
    """
    nan_feature_names = [
        "HRV_LF",
        "HRV_LFHF",
        "HRV_LFn",
        "HRV_MSEn",
        "HRV_CMSEn",
        "HRV_RCMSEn",
        "LF_frequency_power",
        "LF_normalized_power",
    ]

    circadian_features = [
        "circadian_decay",
        "circadian_linear",
        "circadian_cosine",
        "timestamp_start",
    ]

    label_names = [
        "Sleep_Stage",
        "Obstructive_Apnea",
        "Central_Apnea",
        "Hypopnea",
        "Multiple_Events",
        "artifact",
    ]
    info_df = pd.read_csv(info_dir)
    all_subjects_fe_df, good_quality_sids = load_data_to_df(
        threshold, quality_df_dir, info_df, features_dir, nan_feature_names, 
        label_names, circadian_features
)
    clean_df, new_features = clean_features(
        all_subjects_fe_df, info_df, nan_feature_names, label_names
    )
    return clean_df, new_features, good_quality_sids


def split_data(new_df, good_quality_sids, features):
    """
    Splits the dataset into a subset with reduced feature set by removing 
    highly correlated features.

    Parameters:
    ----------
    new_df : pandas DataFrame
        The dataframe containing features and labels for all subjects.
    good_quality_sids : list of str
        A list of subject IDs that met the quality score threshold and were 
        included in the analysis.
    features : list
        A list of feature names to consider for correlation analysis and 
        potential removal.

    Returns:
    -------
    SW_df : pandas DataFrame
        The dataframe with reduced features based on correlation analysis.
    final_features : list
        The list of features retained after removing highly correlated ones.
    """
    train_sids = good_quality_sids[:45]

    corr_matrix = new_df.loc[new_df["sid"].isin(train_sids), features].corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))

    # Find features with correlation greater than a threshold (e.g., 0.8 or 0.9)
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    if "ACC_INDEX" in to_drop:
        to_drop.remove("ACC_INDEX")

    # Drop highly correlated features
    df_reduced = new_df.drop(to_drop, axis=1)

    final_features = [f for f in features if f in df_reduced.columns]

    SW_df = df_reduced.copy()

    return SW_df, final_features


def train_test_split(SW_df, sids, features, group_variable):
    """
    Splits the dataset into features (X), labels (y), and group variables 
    for a specified list of subjects.

    Parameters:
    ----------
    SW_df : pandas DataFrame
        The dataframe with reduced features based on correlation analysis.
    sids : list of strings
        A list of subject IDs for which to extract the data.
    features : list of strings
        A list of feature names to be included in the features array (X).
    group_variable: list
        A list containing the selected variable(s).

    Returns:
    -------
    X : numpy array
        The features array for the specified subjects.
    y : numpy array
        The labels array for the specified subjects.
    group : numpy array
        The group variable array for the specified subjects.
    """
    X = SW_df.loc[SW_df["sid"].isin(sids), features].to_numpy()
    y = SW_df.loc[SW_df["sid"].isin(sids), "Sleep_Stage"].to_numpy()
    group = SW_df.loc[SW_df["sid"].isin(sids), group_variable].to_numpy()
    return X, y, group


def resample_data(X_train, y_train, group_train, group_variable):
    """
    Applies SMOTE resampling to balance the dataset across the target classes.

    Parameters:
    ----------
    X_train : numpy array
        The training features before resampling.
    y_train : numpy array
        The training labels before resampling.
    group_train : numpy array
        The group variable(s) associated with `X_train`.
    group_variable: list
        A list containing the selected variable(s).

    Returns:
    -------
    X_train_resampled : numpy array
        The features after SMOTE resampling.
    y_train_resampled : numpy array
        The labels after SMOTE resampling.
    group_train_resampled : numpy array
        The group variable(s) after SMOTE resampling.
    """
    smote = SMOTE(random_state=1)
    combined = np.column_stack((X_train, group_train))
    combined_resampled, y_train_resampled = smote.fit_resample(combined, y_train)

    # Separate the features and the group variable after resampling
    X_train_resampled = combined_resampled[
        :, : -len(group_variable)
    ]  # All columns except the last one/two, depends on group variable
    group_train_resampled = combined_resampled[:, -len(group_variable) :]

    return X_train_resampled, y_train_resampled, group_train_resampled
