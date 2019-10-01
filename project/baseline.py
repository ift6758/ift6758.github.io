import pandas as pd
import numpy as np

train_profiles_path = "./Profiles.csv"
test_profiles_path = "test_Profiles.csv"

ds = pd.read_csv(train_profiles_path)
print(ds)