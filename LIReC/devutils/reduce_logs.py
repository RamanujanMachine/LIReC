from os import listdir, remove
from os.path import isfile, join

LOGS_PATH = '../logs'

def interesting_line(line):
    return ' - job_logger - job_poly_pslq.py - DEBUG - checking consts: ' not in line

def main():
    for log in [f for f in listdir(LOGS_PATH) if isfile(join(LOGS_PATH, f)) and '.log.' in f]:
        newlines = []
        with open(join(LOGS_PATH, log)) as f: # first recover interesting lines
            last = ''
            for line in f:
                if interesting_line(line):
                    if last:
                        newlines += [last]
                        last = ''
                    newlines += [line]
                else:
                    last = line
            if not newlines:
                newlines = [last]
            f.close()
            remove(join(LOGS_PATH, log))
        with open(join(LOGS_PATH, log), 'w') as f: # then write them down
            f.writelines(newlines)

if __name__ == '__main__':
    main()
