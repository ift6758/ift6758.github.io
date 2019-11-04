import textwrap
from dataclasses import InitVar, dataclass, field

import numpy as np


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


@dataclass
class User():
    userid: str
    gender: str = field(init=False)
    is_female: InitVar[bool] = True
    age_group: str = field(init=False)
    age_group_id: InitVar[int] = 0
    ope: float = 3.91
    con: float = 3.45
    ext: float = 3.49
    agr: float = 3.58
    neu: float = 2.73
    
    def to_xml(self):
        return textwrap.dedent(f"""\
        <user
            id="{self.userid}"
            age_group="{self.age_group}"
            gender="{self.gender}"
            extrovert="{self.ext:.3f}"
            neurotic="{self.neu:.3f}"
            agreeable="{self.agr:.3f}"
            conscientious="{self.con:.3f}"
            open="{self.ope:.3f}"
        />""")
    
    def __post_init__(self, is_female = True, age_group_id = 0):
        self.gender = gender_string(is_female)
        self.age_group = age_group_string(age_group_id)
        self.ope = round(float(self.ope), 3)
        self.con = round(float(self.con), 3)
        self.ext = round(float(self.ext), 3)
        self.agr = round(float(self.agr), 3)
        self.neu = round(float(self.neu), 3)

average_user = User(userid="average_user")
