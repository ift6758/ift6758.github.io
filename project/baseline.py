import pandas as pd
import numpy as np
import argparse
from user import age_group_string, User
import os

def baseline(input_dir: str, output_dir: str, train_profiles_file_path: str):
    """
    input_dir: Input test directory. It is assumed that the folder structure is the same as the Trainign structure.
    output_dir: Output directory, where we will place the xml files for each user.
    train_profiles_file_path: path of the training Profiles.csv file.
    """
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
 
    ds = pd.read_csv(train_profiles_file_path)
    # Drop all rows with 'NA' values, if any.
    ds.dropna(inplace=True)

    # add the age_group column.
    ds = ds.assign(age_group = lambda dt: pd.Series([age_group_string(age) for age in dt["age"]]))
    print(ds)
    # show a description of the dataset.
    print(ds.describe())

    # find the most frequent age group
    most_frequent_age_group_series = ds.age_group.value_counts(sort=True, ascending=False, dropna=False)
    print("Persons per age group:")
    print(most_frequent_age_group_series)
    
    # take the name of the first row in the series
    most_frequent_age_group = next(age_group for age_group, count in most_frequent_age_group_series.items())
    print("Most frequent age group:", most_frequent_age_group)
    average_gender_value = ds.gender.mean()
    most_common_gender = round(average_gender_value)
    print("Average gender value:", average_gender_value)
    print("Most common gender:", most_common_gender)
    
    average_ope = ds.ope.mean()
    average_con = ds.con.mean()
    average_ext = ds.ext.mean()
    average_agr = ds.agr.mean()
    average_neu = ds.neu.mean()
    print("average ope:", average_ope)
    print("average con:", average_con)
    print("average ext:", average_ext)
    print("average agr:", average_agr)
    print("average neu:", average_neu)
    
    # get the list of userid's.
    image_filenames = os.listdir(os.path.join(input_dir, "Image"))
    test_userids = [image_filename.split(".")[0] for image_filename in image_filenames]
    
    # create the 'average' user.
    average_user = User(
        userid='doesnt matter', # we are going to overwrite this value below.
        age=-1, # we don't use the age, just the age group.
        age_group=most_frequent_age_group,
        gender=most_common_gender,
        ope=average_ope,
        con=average_con,
        ext=average_ext,
        agr=average_agr,
        neu=average_neu
    )
    
    for userid in test_userids:
        # modify the userid
        average_user.userid = userid
        with open(os.path.join(output_dir, f"{userid}.xml"), "w") as xml_file:
            xml_file.write(average_user.to_xml())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, default="./debug_data", help="Input directory")
    parser.add_argument("-o", type=str, default="./debug_output", help="Output directory")
    parser.add_argument("--train_profiles_file_path", type=str, default="./debug_profile.csv",
        help="""
            Path to the training 'Profile.csv' file used to compute the baseline statistics.
            When running on the server, this should be set to '/home/mila/teaching/user07/Train/Profile/Profile.csv'
        """
    )
    
    args = parser.parse_args()
    baseline(args.i, args.o, args.train_profiles_file_path)
