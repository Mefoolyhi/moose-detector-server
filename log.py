import datetime

filename = 'log.txt'


def log(log):
    with open(filename, 'a') as f:
        f.write(str(datetime.datetime.now()) + '\n' + str(log) + '\n')

