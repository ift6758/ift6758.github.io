from dataclasses import dataclass
import textwrap


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
    age: float
    gender: float
    ope: float
    con: float
    ext: float
    agr: float
    neu: float
    age_group: str = ""

    def to_xml(self):
        return textwrap.dedent(f"""\
        <user
            id="{self.userid}"
            age_group="{self.age_group}"
            gender="{self.gender_string}"
            extrovert="{self.ext}"
            neurotic="{self.neu}"
            agreeable="{self.agr}"
            conscientious="{self.con}"
            open="{self.ope}"
        />""")
    
    def __post_init__(self):
        if not self.age_group:
            self.age_group = age_group_string(self.age)

    @property 
    def gender_string(self) -> str:
        assert self.gender in (0.0, 1.0), "gender should only be 0.0 or 1.0"
        if self.gender == 0.:
            return "male"
        else:
            return "female"
