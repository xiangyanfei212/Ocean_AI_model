import datetime

def get_date(year, day):
    year = int(year)
    day = int(day)
    first_day = datetime.datetime(year, 1, 1)
    wanted_day = first_day + datetime.timedelta(day)
    wanted_day = datetime.datetime.strftime(wanted_day, "%Y%m%d")
    return wanted_day
