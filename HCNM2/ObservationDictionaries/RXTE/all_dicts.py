# Script to import ALL RXTE observation dictionaries

from ObservationDictionaries.RXTE.dict_40805 import dict_40805
from ObservationDictionaries.RXTE.dict_50098 import dict_50098
from ObservationDictionaries.RXTE.dict_50099 import dict_50099
from ObservationDictionaries.RXTE.dict_60079 import dict_60079
from ObservationDictionaries.RXTE.dict_91802 import dict_91802

all_dicts = [dict_91802, dict_60079, dict_40805, dict_50099, dict_50098]

# Note that observation 50098 has the lock-on problem, so it is not included