#!/usr/bin/env python3

import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd
import textwrap
from dataclasses import dataclass, field, InitVar

from user import User, average_user

def get_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="",
        epilog="""
        """)

    parser.add_argument(
        "-i", "--idir",
        required=True, nargs="+",
        help="")

    parser.add_argument(
        "-o", "--odir",
        required=True, nargs="+",
        help="")

    return parser.parse_args()

def get_sub_ids(input_dir):
    pic_list = glob.glob(os.path.join(input_dir, 'Image', '*.jpg'))
    test_userids = [os.path.basename(pic).split(".")[0] for pic in pic_list]
    return test_userids


def gender_string(gender: bool) -> str:
    if gender:
        return "female"
    else:
        return "male"


def age_group_string(age_group_id: int) -> str:
    """returns the string for the age group:
    either "xx-24", "25-34", "35-49", or "50-xx"
    """
    age_group_strings = ["xx-24", "25-34", "35-49", "50-xx"]
    return age_group_strings[age_group_id]


def get_gender_from_facial_hair(data_dir: str, threshold: float = 0.25) -> pd.DataFrame:
    """Simple baseline that uses the facial hair features to determine gender.
    
    Arguments:
        data_dir {str} -- The input data directory
    
    Keyword Arguments:
        threshold {float} -- Threshold to use. If the sum of the "beard" and "mustache" is smaller than the threshold, the user is considered female (default: {0.25})
    
    Returns:
        pd.DataFrame -- dataframe with userids and their associated gender, as a boolean (True for Female, False for Male)
    """
    liwc_path = os.path.join(data_dir, "Text", "liwc.csv")
    nrc_path = os.path.join(data_dir, "Text", "nrc.csv")
    likes_path = os.path.join(data_dir, "Relation", "Relation.csv")
    oxford_path = os.path.join(data_dir, "Image", "oxford.csv")
    
    #TODO PATH_IMAGE + PATH_PROFILES
    #profiles = pd.read_csv(PATH_PROFILES)
    
    #profiles = pd.read_csv(PATH_PROFILES)
    liwc = pd.read_csv(liwc_path)
    nrc = pd.read_csv(nrc_path)
    likes = pd.read_csv(likes_path)
    oxford = pd.read_csv(oxford_path)

    liwc = liwc.rename(columns={"userId":"userid"})
    oxford = oxford.rename(columns={"userId":"userid"})
    nrc = nrc.rename(columns={"userId":"userid"})
    likes = likes.rename(columns={"userId":"userid"})        

    user_ids = liwc.merge(oxford["userid"], on="userid", how='outer')
    user_ids = user_ids.merge(nrc["userid"], on="userid", how='outer')
    user_ids = user_ids.merge(likes["userid"], on="userid", how='outer')
    user_ids=user_ids.loc[:,'userid'].unique()

    oxford = oxford.rename(columns={"userId":"userid"})
    oxford.drop_duplicates(subset ="userid",keep = "first", inplace=True)

    def get_amount_of_hair(oxford: pd.DataFrame) -> pd.Series:
        return oxford.facialHair_mustache + oxford.facialHair_beard
    
    facial_hair = oxford.loc[:,['userid']]
    facial_hair['hair'] = get_amount_of_hair(oxford)

    women = facial_hair.loc[:,['userid']]
    women['gender'] = facial_hair['hair'] < threshold
    
    women.set_index("userid", inplace=True)
    return women


def main():
    args = get_arguments()
    output_dir = args.odir[0]
    input_dir = args.idir[0]

    # Create oFolder if not exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    gender_dataframe = get_gender_from_facial_hair(input_dir)
    # TODO: create the predictions from the model here, like in test.py.

    for (userid, is_female) in gender_dataframe.itertuples():
        user = User(
            userid = userid,
            is_female=gender_dataframe.gender[userid],
            ope = 3.91,
            con = 3.45,
            ext = 3.49,
            agr = 3.58,
            neu = 2.73,
        )
        print(user)
        with open(os.path.join(output_dir, f"{userid}.xml"), "w") as xml_file:
            xml_file.write(user.to_xml())

if __name__ == '__main__':
    sys.exit(main())
