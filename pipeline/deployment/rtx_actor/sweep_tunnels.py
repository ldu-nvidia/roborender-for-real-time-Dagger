"""Own four independent loopback-only forwards for the isolated H200 sweep."""
from concurrent.futures import ThreadPoolExecutor
import argparse
from pathlib import Path
import socket
import subprocess

CONFIG = Path('/home/ldu/.ssh/roborender_h200_sweep_1009_config')
CONTROL = Path('/home/ldu/.ssh/roborender-sweep-1009')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'stop'])
    args = parser.parse_args()
    if socket.gethostname() != '465d6b0-lcedt' or not CONFIG.is_file():
        raise RuntimeError('wrong host or missing pinned SSH configuration')
    CONTROL.mkdir(mode=0o700, exist_ok=True)
    ports = [18620, 18621, 18622, 18623]
    if args.action == 'start':
        for port in ports:
            with socket.socket() as test:
                test.bind(('127.0.0.1', port))
            if (CONTROL / str(port)).exists():
                raise RuntimeError('refusing to replace an existing control socket')

    def run(port):
        base = ['ssh', '-F', str(CONFIG), '-S', str(CONTROL / str(port))]
        if args.action == 'start':
            command = base + ['-M', '-fNT', '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes',
                              '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
                              '-L', f'127.0.0.1:{port}:127.0.0.1:{port}', 'roborender-h200-sweep']
        else:
            command = base + ['-O', 'exit', 'roborender-h200-sweep']
        # A background SSH proxy can inherit stderr. Do not use communicate()
        # pipes here: waiting for their EOF waits for the entire tunnel lifetime.
        log_path = CONTROL / f'{port}.{args.action}.log'
        with log_path.open('ab') as log:
            result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=log, timeout=90)
        return port, result.returncode, str(log_path)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, ports))
    for port, code, error in results:
        print(f'{args.action} port={port} returncode={code} {error}', flush=True)
    if args.action == 'start' and any(code for _, code, _ in results):
        for port, code, _ in results:
            if not code:
                subprocess.run(['ssh', '-F', str(CONFIG), '-S', str(CONTROL / str(port)),
                                '-O', 'exit', 'roborender-h200-sweep'], timeout=10)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
