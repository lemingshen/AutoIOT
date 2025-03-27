import sys
import subprocess


def run(cmd, shell=True):
    """
    create a subprocess, use the shell to execute the command
    store the console output of the execution process in 'result'
    return the state code and the console output

    :param cmd: command of the subprocess
    :param shell: use shell or not
    :return: status code and console output
    """
    print('\033[1;32m************** START **************\033[0m')
    p = subprocess.Popen(cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    result = []
    while p.poll() is None:
        line = p.stdout.readline().strip()
        if line:
            line = _decode_data(line)
            result.append(line)
            print('\033[1;35m{0}\033[0m'.format(line))
        
        # clear cache
        sys.stdout.flush()
        sys.stderr.flush()
    
    if p.returncode == 0:
        print('\033[1;32m************** SUCCESS **************\033[0m')
    else:
        print('\033[1;31m************** FAILED **************\033[0m')
    return p.returncode, '\r\n'.join(result)


def _decode_data(byte_data: bytes):
    try:
        return byte_data.decode('UTF-8')
    except UnicodeDecodeError:
        return byte_data.decode('GB18030')


if __name__ == '__main__':
    return_code, data = run('python3 test.py')
    print('return code:', return_code, 'data:', data)

