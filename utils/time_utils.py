import datetime

def datetime_to_str(datetime: datetime.datetime):
  datetime_str = datetime.strftime('%d%m%y_%H:%M')
  return datetime_str

def now_to_str():
  now = datetime.datetime.now()
  now_str = now.strftime('%d%m%y_%H:%M')
  return now_str

def str_to_datetime(string: str):
  datetime = datetime.strptime(string, '%d%m%y_%H:%M')
  return datetime

# Pandas stores datetimes in different format and this converts the datetime to our format
def pd_datatime_str_to_str(string: str) -> str:
  datetime = datetime.strptime(string, '%Y-%m-%d %H:%M:%S.%f')
  datetime_str = datetime.strftime('%d%m%y_%H:%M')

