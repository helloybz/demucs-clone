import subprocess as sp
import sys
import time
import socket
import random
import errno

import torch


def free_port(host='', low=20000, high=40000):
    sock = socket.socket()
    while True:
        port = random.randint(low, high)
        try:
            sock.bind((host, port))
        except OSError as error:
            if error.errno == errno.EADDRINUSE:
                continue
            raise
        return port


def run_with_gpu(args):
    gpus = torch.cuda.device_count()

    def free_port(host='', low=20000, high=40000):
        sock = socket.socket()
        while True:
            port = random.randint(low, high)
            try:
                sock.bind((host, port))
            except OSError as error:
                if error.errno == errno.EADDRINUSE:
                    continue
                raise
            return port

    port = free_port()
    args += ["--world_size", str(gpus), "--master_url", f"127.0.0.1:{port}"]
    tasks = []

    for gpu in range(gpus):
        kwargs = {}
        if gpu > 0:
            kwargs['stdin'] = sp.DEVNULL
            kwargs['stdout'] = sp.PIPE
            # We keep stderr to see tracebacks from children.
        tasks.append(sp.Popen(["python3", "-m", "demucs_clone"] + args + ["--rank", str(gpu)], **kwargs))
        tasks[-1].rank = gpu

    failed = False
    try:
        while tasks:
            for task in tasks:
                if task.stdout:
                    print(task.stdout.read().decode('utf-8'))
                try:
                    exitcode = task.wait(0.1)
                except sp.TimeoutExpired:
                    continue
                else:
                    tasks.remove(task)
                    if exitcode:
                        print(f"Task {task.rank} died with exit code "
                              f"{exitcode}",
                              file=sys.stderr)
                        failed = True
            if failed:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        for task in tasks:
            task.terminate()
        raise
    if failed:
        for task in tasks:
            task.terminate()
        sys.exit(1)


def run_with_cpu(args):
    task = sp.Popen(["python3", "-m", "demucs_clone"] + args, stdout=sys.stdout)
    failed = False
    try:
        while True:
            try:
                exitcode = task.wait(0.1)
            except sp.TimeoutExpired:
                continue
            else:
                if exitcode:
                    print(f"The task died with exit code "
                          f"{exitcode}",
                          file=sys.stderr)
                break
        time.sleep(1)
    except KeyboardInterrupt:
        task.terminate()
    if failed:
        task.terminate()
        sys.exit(1)


def main():
    args = sys.argv[1:]

    if torch.cuda.is_available():
        run_with_gpu(args)
    else:
        run_with_cpu(args)


if __name__ == "__main__":
    main()
