from LIReC.jobs import run
import sys

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Only commands are start and stop')
        exit(1)

    if sys.argv[1] == 'stop':
        run.stop()
    elif sys.argv[1] == 'start':
        run.main()
    else:
        print('Only commands are start and stop')
        exit(1)
