import os
import glob

import numpy as np
import pandas as pd

import tensorflow as tf

from typing import *
from collections import Counter
from utils import DEBUG

def get_text_data(input_dir):
    """
    Purpose: preprocess liwc and nrc
    Input
        input_dir {string} : path to input_directory (ex, "~/Train")
    Output:
        id_list {numpy array of strings}: array of user ids sorted alphabetically,
                                        to determine order of features and labels DataFrames
        text_data {pandas DataFrame of float}: unscaled text data (liwc and nrc combined)
    """
    # Load and sort text data
    liwc = pd.read_csv(os.path.join(input_dir, 'Text/liwc.csv'), sep = ',')
    liwc = liwc.sort_values(by=['userId'])

    nrc = pd.read_csv(os.path.join(input_dir, 'Text/nrc.csv'), sep = ',')
    nrc = nrc.sort_values(by=['userId'])

    # Build list of subject ids ordered alphabetically
    # Check if same subject lists in both sorted DataFrames (liwc and nrc)
    if np.array_equal(liwc['userId'], nrc['userId']):
        id_list = liwc['userId'].to_numpy()
    else:
        raise Exception('userIds do not match between liwc and nrc data')

    # merge liwc and nrc DataFrames using userId as index
    liwc.set_index('userId', inplace=True)
    nrc.set_index('userId', inplace=True)

    text_data = pd.concat([liwc, nrc], axis=1, sort=False)

    return id_list, text_data


def get_image_clean(sub_ids, oxford, means):
    '''
    Purpose: preprocess oxford metrics derived from profile pictures (part 2)
    Input:
        sub_ids {numpy array of strings}: ordered list of userIDs
        oxford {pandas DataFrame of floats}: unscaled oxford features of users with 1+ face
        means {list of float}: mean values for each feature averaged from train set,
                    to replace missing values for userids with no face (train and test set)
    Output:
        image_data {pandas DataFrame of float}: unscaled oxford image data
                with mean values replacing missing entries
    '''
    # list of ids with at least one face on image: 7174 out of 9500 in train set
    ox_list = np.sort(oxford['userId'].unique(), axis=None)
    # list of ids in text_list who have no face metrics in oxford.csv (2326 in train set)
    ox_noface = np.setdiff1d(sub_ids, ox_list)

    # Create DataFrame for userids with no face (1 row per userid)
    # values are mean metrics averaged from users with entries (training set)
    ox_nf = pd.DataFrame(ox_noface, columns = ['userId'])
    columns = oxford.columns[2:].tolist()
    for column, mean in zip(columns, means):
        ox_nf.insert(loc=ox_nf.shape[1], column=column, value=mean, allow_duplicates=True)
    # insert column 'noface' = 1 if no face in image, else 0
    ox_nf.insert(loc=ox_nf.shape[1], column='noface', value=1, allow_duplicates=True)
    # insert column 'multiface' = 1 if many faces in image, else 0
    ox_nf.insert(loc=ox_nf.shape[1], column='multiface', value=0, allow_duplicates=True)
    ox_nf.set_index('userId', inplace=True)

    # Format DataFrame from userids with 1+ face
    # insert column 'noface' = 1 if no face in image, else 0
    oxford.insert(loc=oxford.shape[1], column='noface', value=0, allow_duplicates=True)
    # list userIds with multiple faces (714 in train set)
    ox_multiples = oxford['userId'][oxford['userId'].duplicated()].tolist()
    # insert column 'multiface' = 1 if many faces in image, else 0
    oxford.insert(loc=oxford.shape[1], column='multiface', value=0, allow_duplicates=True)
    multi_mask = pd.Series([uid in ox_multiples for uid in oxford['userId']])
    i = oxford[multi_mask].index
    oxford.loc[i, 'multiface'] = 1
    # drop duplicate entries with same userId (keep first entry per userId)
    oxford.drop_duplicates(subset ='userId', keep='first', inplace=True)

    # merge the two DataFrames
    oxford.drop(['faceID'], axis=1, inplace=True)
    oxford.set_index('userId', inplace=True)
    image_data = pd.concat([ox_nf, oxford], axis=0, sort=False).sort_values(by=['userId'])

    if not np.array_equal(image_data.index, sub_ids):
        raise Exception('userIds do not match between oxford file and id list')

    return image_data


def get_image_raw(data_dir):
    '''
    Purpose: preprocess oxford metrics derived from profile pictures (part 1)
    Input
        input_dir {string} : path to input_directory (ex, "~/Train")
    Output:
        image_data {pandas DataFrame of float}: unscaled oxford image data
    '''
    # Load data of oxford features extracted from profile picture (face metrics)
    # 7915 entries; some users have no face, some have multiple faces on image.
    # userids with 1+ face on image: 7174 out of 9500 (train set)
    # duplicated entries (userids with > 1 face on same image): 741 in train set
    oxford = pd.read_csv(os.path.join(data_dir, "Image", "oxford.csv"), sep = ',')
    oxford = oxford.sort_values(by=['userId'])
    '''
    NOTE: headPose_pitch has NO RANGE, drop that feature
    '''
    oxford.drop(['headPose_pitch'], axis=1, inplace=True)

    return oxford


def get_likes_kept(data_dir, num_features) -> List[str]:
    '''
    Purpose: get list of likes to keep as features
    Input:
        data_dir {str} : the parent input directory
        num_features {int} : the number of likes to keep as features,
                        starting from those with highest frequencies
    Output:
        freq_like_id {List of strings}: frequency of most frequent likes,
                    (number = num_features), in descending ordered, indexed by like_id
    '''
    #Why return frequency?
    relation = pd.read_csv(os.path.join(data_dir, "Relation", "Relation.csv")) #, index_col=1)
    relation = relation.drop(['Unnamed: 0'], axis=1)
    like_ids_to_keep = relation['like_id'].value_counts(sort=True, ascending=False)[:num_features] #This sorts features by frequency

    #sort like indices (which are the keys associated with the values kepts)
    likes_int64_list = sorted(like_ids_to_keep.keys()) # This sorts indices by like_id
    likes_str_list = [str(l) for l in likes_int64_list]
    return likes_str_list


def get_relations(data_dir: str, sub_ids: List[str], like_ids_to_keep: List[str]):
    '''
    Purpose: preprocess relations dataset ('likes')

    Input:
        data_dir {str} -- the parent input directory
        sub_ids {numpy array of strings} -- the ordered list of userids
        like_ids_to_keep {List[str]} -- The list of page IDs to keep.

    Returns:
        relations_data -- multihot matrix of the like_id. Rows are indexed with userid, entries are boolean.
    '''
    relation = pd.read_csv(os.path.join(data_dir, "Relation", "Relation.csv")) #, index_col=1)
    relation = relation.drop(['Unnamed: 0'], axis=1)

    ## One HUGE step:
    # likes_to_keep = like_ids_to_keep.keys()
    # kept_relations = relation[relation.like_id.isin(likes_to_keep)]
    # multi_hot_relations = pd.get_dummies(kept_relations, columns=["like_id"], prefix="")
    # multi_hot = multi_hot_relations.groupby(("userid")).sum()
    # return multi_hot_relations
    ###
    total_num_pages = len(like_ids_to_keep)
    # Create a multihot likes matrix of booleans (rows = userids, cols = likes), by batch
    batch_size = 1000
    
    # Create empty DataFrame with sub_ids as index list
    relation_data = pd.DataFrame(sub_ids, columns = ['userid'])
    relation_data.set_index('userid', inplace=True)

    for start_index in range(0, total_num_pages, batch_size):
        end_index = min(start_index + batch_size, total_num_pages)

        # sets are better for membership testing than lists. 
        like_ids_for_this_batch = set(like_ids_to_keep[start_index:end_index])

        filtered_table = relation[relation['like_id'].isin(like_ids_for_this_batch)]
        ## THIS is the slow part:
        relHot = pd.get_dummies(filtered_table, columns=['like_id'], prefix="", prefix_sep="")
        ##
        relHot = relHot.groupby(['userid']).sum().astype(float) # this makes userid the index
        
        relation_data = pd.concat([relation_data, relHot], axis=1, sort=True)
    
    relation_data = relation_data.reindex(like_ids_to_keep, axis=1)
    relation_data.fillna(0.0, inplace=True)
    relation_data = relation_data.astype("bool")
    
    # will be different if users in relation.csv are not in sub_ids
    if not np.array_equal(relation_data.index, sub_ids):
        raise Exception(f"""userIds do not match between relation file and id list:
    {relation_data.index}
    {sub_ids}
    
    """)
      
    return relation_data


def make_label_dict(labels):
    '''
    Purpose: make dictionnary of labels from pandas DataFrame
    Input:
        labels {pandas DataFrame}: labels ordered per userids (alphabetical order)
    Output:
        labels_dict {dictionary of pandas DataFrames}: labels (one entry per metric) ordered alphabetically
                by userid for the training set, with userids as index.

    '''
    gender = labels['gender']

    age_grps = labels[['age_xx_24', 'age_25_34', 'age_35_49', 'age_50_xx']]

    '''
    Note: : each DataFrames (value) is indexed by userid in labels_dict
    '''
    labels_dict = {}
    labels_dict['userid'] = labels.index
    labels_dict['gender'] = gender
    labels_dict['age_grps'] = age_grps
    labels_dict['ope'] = labels['ope']
    labels_dict['con'] = labels['con']
    labels_dict['ext'] = labels['ext']
    labels_dict['agr'] = labels['agr']
    labels_dict['neu'] = labels['neu']

    return labels_dict


def preprocess_labels(data_dir, sub_ids):
    '''
    Purpose: preprocess entry labels from training set
    Input:
        datadir {string} : path to training data directory
        sub_ids {numpy array of strings}: list of subject ids ordered alphabetically
    Output:
        labels {pandas DataFrame}: labels ordered by userid (alphabetically)
                for the training set, with userids as index.

    '''
    labels = pd.read_csv(os.path.join(data_dir, "Profile", "Profile.csv"))
    labels = labels.sort_values(by=['userid'])
    # check if same subject ids in labels and sub_ids
    if not np.array_equal(labels['userid'].to_numpy(), sub_ids):
        raise Exception('userIds do not match between profiles labels and id list')

    def age_group_id(age_str: str) -> int:
        """Returns the age group category ID (an integer from 0 to 3) for the given age (string)
        
        Arguments:
            age_str {str} -- the age
        
        Returns:
            int -- the ID of the age group: 0 for xx-24, 1 for 25-34, 2 for 35-49 and 3 for 50-xx.
        """
        age = int(age_str)
        if age <= 24:
            return 0
        elif age <= 34:
            return 1
        elif age <= 49:
            return 2
        else:
            return 3

    labels = labels.assign(age_group = lambda dt: pd.Series([age_group_id(age_str) for age_str in dt["age"]]))
    # labels = labels.assign(age_xx_24 = lambda dt: pd.Series([int(age) <= 24 for age in dt["age"]]))
    # labels = labels.assign(age_25_34 = lambda dt: pd.Series([25 <= int(age) <= 34 for age in dt["age"]]))
    # labels = labels.assign(age_35_49 = lambda dt: pd.Series([35 <= int(age) <= 49 for age in dt["age"]]))
    # labels = labels.assign(age_50_xx = lambda dt: pd.Series([50 <= int(age) for age in dt["age"]]))

    labels = labels.drop(['Unnamed: 0'], axis=1)
    labels.set_index('userid', inplace=True)

    return labels


def preprocess_train(data_dir, num_likes=10_000):
    '''
    Purpose: preprocesses training dataset (with labels) and returns scaled features,
    labels and parameters to scale the test data set
    Input
        data_dir {string}: path to ~/Train data directory
        num_likes {int}: number of like_ids to keep as features
    Output:
        train_features {pandas DataFrame}: vectorized features scaled between 0 and 1
                for each user id in the training set, concatenated for all modalities
                (order = text + image + relation), with userid as DataFrame index.
        features_min_max {tupple of 2 pandas Series}: series of min and max values of
                text + image features from train dataset, to be used to scale test data.
                Note that the multihot relation features do not necessitate scaling.
        image_means {list of float}: means from oxford dataset to replace missing entries in oxford test set
        likes_kept {list of strings}: ordered likes_ids to serve as columns for test set relation features matrix
        train_labels {pandas DataFrame}: labels ordered by userid (alphabetically)
                for the training set, with userids as index.

    TO CONSIDER: convert outputted pandas to tensorflow tf.data.Dataset...
    https://www.tensorflow.org/guide/data
    '''
    # sub_ids: a numpy array of subject ids ordered alphabetically.
    # text_data: a pandas DataFrame of unscaled text data (liwc and nrc)
    sub_ids, text_data = get_text_data(data_dir)
    # image_data: pandas dataframe of oxford data
    # image_min_max: a tupple of 2 pandas series, the min and max values from oxford training features
    image_data_raw = get_image_raw(data_dir)
    image_means = image_data_raw.iloc[:, 2:].mean().tolist()
    image_data = get_image_clean(sub_ids, image_data_raw, image_means)

    '''
    Note: Scale the text and image data BEFORE concatenating with relations
    '''
    features_to_scale = pd.concat([text_data, image_data], axis=1, sort=False)
    feat_min = features_to_scale.min()
    feat_max = features_to_scale.max()

    feat_scaled = (features_to_scale - feat_min) / (feat_max - feat_min)
    features_min_max = (feat_min, feat_max)

    if DEBUG:
        likes_kept = [str(v) for v in range(num_likes)]
    else:
        likes_kept = get_likes_kept(data_dir, num_likes)

    # multi-hot matrix of likes from train data
    likes_data = get_relations(data_dir, sub_ids, likes_kept)

    # concatenate all scaled features into a single DataFrame
    train_features = pd.concat([feat_scaled, likes_data], axis=1, sort=False)

    # DataFrame of training set labels
    train_labels = preprocess_labels(data_dir, sub_ids)

    return train_features, features_min_max, image_means, likes_kept, train_labels


def preprocess_test(data_dir, min_max_train, image_means_train, likes_kept_train):
    '''
    Purpose: preprocesses test dataset (no labels)
    Input:
        datadir {string}: path to Test data directory
        min_max_train {tupple of two numpy arrays}: min and max values for
                concatenated text and image features (from train set)
        image_means_train {list of float}: means from oxford training dataset to replace
                missing entries in oxford test set
        likes_kept_train {list of strings}: most frequent likes_ids from train set
                (ordered by frequency) to serve as columns in relation features matrix
    Output:
        test_features {pandas DataFrame}: vectorized features of test set
    '''
    # sub_ids: a numpy array of subject ids ordered alphabetically.
    # text_data: a pandas DataFrame of unscaled text data (liwc and nrc)
    sub_ids, text_data = get_text_data(data_dir)

    # image_data: pandas dataframe of oxford data
    # image_min_max: a tupple of 2 pandas series, the min and max values from oxford training features
    image_data_raw = get_image_raw(data_dir)
    image_data = get_image_clean(sub_ids, image_data_raw, image_means_train)

    '''
    Note: Scale the text and image data BEFORE concatenating with relations
    '''
    features_to_scale = pd.concat([text_data, image_data], axis=1, sort=False)
    feat_min = min_max_train[0]
    feat_max = min_max_train[1]

    feat_scaled = (features_to_scale - feat_min) / (feat_max - feat_min)

    # multi-hot matrix of likes from train data
    likes_data = get_relations(data_dir, sub_ids, likes_kept_train)
    
    # concatenate all scaled features into a single DataFrame
    test_features = pd.concat([feat_scaled, likes_data], axis=1, sort=False)

    return test_features


def get_train_val_sets(features, labels, val_prop):
    '''
    Purpose: Splits training dataset into a train and a validation set of
    ratio determined by val_prop (x = features, y = labels)
    Input
        features {pandas DataFrame}: vectorized features scaled between 0 and 1
                for each user id in the training set, concatenated for all modalities
                (order = text + image + relation), with userid as DataFrame index.
        labels {pandas DataFrame}: labels ordered by userid (alphabetically)
                for the training set, with userids as index.
        val_prop {float between 0 and 1}: proportion of sample in validation set
                    (e.g. 0.2 = 20% validation, 80% training)
    Output:
        x_train, x_val {pandas DataFrames}: vectorized features for train and validation sets
        y_train, y_val {pandas DataFrames}: train and validation set labels

    TO DO: convert outputted pandas to tensorflow tf.data.Dataset?...
    https://www.tensorflow.org/guide/data
    '''
    # NOTE: UNUSED
    from sklearn import model_selection
    x_train, x_val, y_train, y_val = model_selection.train_test_split(
        features, # training features to split
        labels, # training labels to split
        test_size = val_prop, # between 0 and 1, proportion of sample in validation set (e.g., 0.2)
        shuffle= True,
        #stratify = y_data[:1],
        # random_state = 42  # can use to always obtain the same train/validation split
        )

    return x_train, x_val, y_train, y_val
