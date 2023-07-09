import datetime

def date_conversation(year,day):
    #输入的字符串类型的年和日转换为整型
    year=int(year)
    day=int(day)
    #first_day：此年的第一天
    #类型：datetime
    first_day=datetime.datetime(year,1,1)
    #用一年的第一天+天数-1，即可得到我们期望的日期
    #-1是因为当年的第一天也算一天
    wanted_day=first_day+datetime.timedelta(day-1)
    #返回需要的字符串形式的日期
    wanted_day=datetime.datetime.strftime(wanted_day,'%Y%m%d')
    return wanted_day

if __name__ == '__main__':
    date_str = date_conversation(2012, 321)
    print(date_str)
