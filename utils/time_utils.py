import datetime

def now_to_str():
  now = datetime.datetime.now()
  now_str = now.strftime('%d%m%y_%H:%M')
  return now_str

def str_to_datetime(string):
  datetime = datetime.strptime(string, '%d%m%y_%H:%M')
  return datetime
