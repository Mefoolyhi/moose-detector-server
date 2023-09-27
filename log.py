import datetime

filename = 'log.txt'


def log(log):
    with open(filename, 'w') as f:
        f.write(str(datetime.datetime.now()) + '\n' + log + '\n')

