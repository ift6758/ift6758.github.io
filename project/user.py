from dataclasses import dataclass, field
import textwrap
import numpy as np

def age_group_string(age: float) -> str:
    """returns the string for the age group:
    either "xx-24", "25-34", "35-49", or "50-xx"
    """
    age_int = int(age)
    if age_int <= 24:
        return "xx-24"
    elif 25 <= age_int <= 34:
        return "25-34"
    elif 35 <= age_int <= 49:
        return "35-49"
    else:
        return "50-xx"

@dataclass
class User():
    userid: str
    gender: int
    ope: float
    con: float
    ext: float
    agr: float
    neu: float
    age_group_id: int
    age_group_string: str = field(init=False)

    def to_xml(self):
        return textwrap.dedent(f"""\
        <user
            id="{self.userid}"
            age_group="{self.age_group_string}"
            gender="{self.gender_string}"
            extrovert="{self.ext}"
            neurotic="{self.neu}"
            agreeable="{self.agr}"
            conscientious="{self.con}"
            open="{self.ope}"
        />""")
    
    def __post_init__(self):
        self.age_group_string = age_group_string(self.age_group_id)

        self.ope = float(self.ope)
        self.con = float(self.con)
        self.ext = float(self.ext)
        self.agr = float(self.agr)
        self.neu = float(self.neu)

    @property 
    def gender_string(self) -> str:
        assert self.gender in (0, 1), "gender should only be 0.0 or 1.0"
        if self.gender == 0:
            return "male"
        else:
            return "female"
