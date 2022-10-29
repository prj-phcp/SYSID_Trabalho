from datetime import datetime
m_2_number = {
    'jan': 1,
    'fev': 2,
    'mar': 3,
    'abr': 4,
    'mai': 5,
    'jun': 6,
    'jul': 7,
    'ago': 8,
    'set': 9,
    'out': 10,
    'nov': 11,
    'dez': 12
}
def convert_dates(strdate):
    d, m, y = strdate.split('-')

    return datetime(2000 + int(y), m_2_number[m], int(d))
