#! /usr/bin/python3

import datetime
import itertools
import math
import os
import re
import subprocess
import time
from sys import argv
from urllib.parse import quote


# --- configurations of experiment hosts (feel free to modify these configurations) ---


conf = {
    # number of physical cores per host
    'maxcore': 24,

    # hosts for hybrid simulation (they should be able to log in via ssh without password)
    'hosts': [
        '172.16.0.2',
        '172.16.0.3',
        '172.16.0.4',
        '172.16.0.5',
        '172.16.0.6',
        '172.16.0.7',
    ],

    # callback when the experiment finished
    'report': lambda msg: print(msg)
}


# --- experiment class definition ---


class ResultCallback:
    def __init__(self, names=[], callback=None):
        self.names = names
        self.callback = callback

    def __call__(self) -> list:
        if callable(self.callback):
            return self.callback()
        else:
            return []

    @staticmethod
    def make(names: list):
        def wrapper(func):
            return ResultCallback(names, func)
        return wrapper


class Experiment:
    def __init__(self, name: str):
        ts = datetime.datetime.now().strftime('%m%d-%H%M')
        self.name = name
        self.cmd_output = open(f'results/{name}-{ts}.sh', 'wt', buffering=1)
        self.raw_output = open(f'results/{name}-{ts}.txt', 'wt', buffering=1)
        self.data_output = open(f'results/{name}-{ts}.csv', 'wt', buffering=1)

    def __del__(self):
        self.cmd_output.close()
        self.raw_output.close()
        self.data_output.close()

    def cmd(self, cmd: list, run=True):
        cmd_str = ' '.join([f'"{c}"' if ' ' in c else c for c in cmd]) + '\n'
        self.cmd_output.write(cmd_str)
        self.raw_output.write(cmd_str)
        if run:
            subprocess.run(cmd)

    def run(self, programs: list, simulators: list, template_cmd='', branch='main', callback=ResultCallback(), example=False, **kwargs):
        # checkout
        self.cmd(['git', 'checkout', branch])

        # write header
        self.data_output.write('program,simulator')
        for arg in sorted(kwargs.keys()):
            self.data_output.write(',' + arg)
        for name in callback.names:
            self.data_output.write(',' + name)
        self.data_output.write(',ret,fct,e2ed,throughput,ev,t\n')

        # enumerate every possible argument permutations
        arg_permutations = []
        for k, v in kwargs.items():
            if isinstance(v, list):
                arg_permutations.append(itertools.product([k], v))
            else:
                arg_permutations.append([(k, v)])

        # run all programs under every argument combinations
        for args in list(itertools.product(*arg_permutations)):
            for program in programs if isinstance(programs, list) else [programs]:
                for simulator in simulators if isinstance(simulators, list) else [simulators]:
                    self.run_once(program, simulator, template_cmd=template_cmd, callback=callback, example=example, **dict([(k, v(dict(args)) if callable(v) else v) for k, v in args]))

        # check back
        self.cmd(['git', 'checkout', branch])

        # push finished notifications
        if 'report' in conf:
            conf['report'](f'Experiment {self.name} finished.')

    def run_once(self, program: str, simulator: str, template_cmd='', callback=ResultCallback(), example=False, **kwargs):
        args = kwargs.copy()
        enable_modules = ['applications', 'csma', 'flow-monitor', 'mpi', 'mtp', 'nix-vector-routing', 'point-to-point']
        configure_cmd = ['./ns3', 'configure', '-d', 'optimized', '--enable-examples', '--enable-modules', ','.join(enable_modules)]
        mpi_cmd = ''

        # configure
        core = args.pop('core', 1)
        if simulator == 'mtp':
            if example:
                program += '-mtp'
            else:
                args['thread'] = core
            configure_cmd.append('--enable-mtp')
        elif simulator == 'barrier':
            args['nullmsg'] = False
            configure_cmd.append('--enable-mpi')
            mpi_cmd = f'mpirun -n {core} --map-by ppr:{conf["maxcore"]}:node --bind-to core'
        elif simulator == 'nullmsg':
            args['nullmsg'] = True
            configure_cmd.append('--enable-mpi')
            mpi_cmd = f'mpirun -n {core} --map-by ppr:{conf["maxcore"]}:node --bind-to core'
        elif simulator == 'hybrid':
            args['thread'] = math.ceil(core / math.ceil(core / conf["maxcore"]))
            configure_cmd.append('--enable-mtp')
            configure_cmd.append('--enable-mpi')
            mpi_cmd = f'mpirun -n {math.ceil(core / conf["maxcore"])} --map-by ppr:1:node --bind-to none'
        if mpi_cmd != '' and core > conf['maxcore']:
            mpi_cmd += f' --host ' + ','.join([f'{h}:{conf["maxcore"]}' for h in conf['hosts']])
        template_cmd = f'{mpi_cmd} {template_cmd}'.strip()

        # build
        self.cmd(['./ns3', 'clean'])
        self.cmd(configure_cmd)
        self.cmd(['./ns3', 'build', program])

        # run
        args_str = ' '.join([f'--{k}={v}' for k, v in args.items()])
        if template_cmd == '':
            run_cmd = ['./ns3', 'run', f'{program} {args_str}'.strip()]
        else:
            run_cmd = ['./ns3', 'run', program, '--command-template', f'{template_cmd} %s {args_str}'.strip()]
        self.cmd(run_cmd, run=False)
        process = subprocess.Popen(run_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        # check event count and simulation time
        ev = 0
        t = 0
        fct = 0
        e2ed = 0
        throughput = 0

        t_start = time.time()
        for line in process.stdout:
            m = re.match(r'  Average flow completion time = (.*)us\n', line)
            if m is not None:
                fct = m.group(1)
            m = re.match(r'  Average end to end delay = (.*)us\n', line)
            if m is not None:
                e2ed = m.group(1)
            m = re.match(r'  Average flow throughput = (.*)Gbps\n', line)
            if m is not None:
                throughput = m.group(1)
            m = re.match(r'  Event count = (.*)\n', line)
            if m is not None:
                ev = m.group(1)
            m = re.match(r'  Simulation time = (.*)s\n', line)
            if m is not None:
                t = m.group(1)
            self.raw_output.write(line)
        process.wait()
        t_end = time.time()
        if t == 0:
            t = t_end - t_start

        # output statistics
        self.data_output.write(f'{program},{simulator}')
        for _, value in sorted(kwargs.items()):
            self.data_output.write(f',{value}')
        for value in callback():
            self.data_output.write(f',{value}')
        self.data_output.write(f',{process.returncode},{fct},{e2ed},{throughput},{ev},{t}\n')

        # push finished notifications
        if 'report' in conf:
            conf['report'](f'{self.name}: {program}-{simulator} returns {process.returncode} in {t} seconds.\n\n{run_cmd}')


# --- experiment result callbacks ---


@ResultCallback.make(['msg', 'sync', 'exec'])
def get_mpi_time() -> list:
    t_msg = 0
    t_sync = 0
    t_exec = 0
    t_sync_list = []
    t_exec_list = []
    system_count = 0
    # calculate average time
    for filename in os.listdir('results'):
        if filename.startswith('BS-') or filename.startswith('NM-'):
            t_sync_round_list = []
            t_exec_round_list = []
            system_count += 1
            fullname = f'results/{filename}'
            with open(fullname, 'rt') as f:
                f.readline()
                line = f.readline().split(',')
                t_msg += int(line[-3])
                t_sync += int(line[-2])
                t_exec += int(line[-1])
                for line in f.readlines():
                    line = line.split(',')
                    t_sync_round_list.append(int(line[-2]))
                    t_exec_round_list.append(int(line[-1]))
            t_sync_list.append(t_sync_round_list)
            t_exec_list.append(t_exec_round_list)
            os.unlink(fullname)
    t_msg /= system_count * 1e9
    t_sync /= system_count * 1e9
    t_exec /= system_count * 1e9
    # output exec time in each round
    with open(f'results/mpi-exec-{system_count}.csv', 'wt') as f:
        f.write('rank,round,exec\n')
        for rank in range(system_count):
            for rnd in range(101):
                try:
                    f.write(f'{rank},{rnd},{sum(t_exec_list[rank][rnd * 100:rnd * 100 + 100])}\n')
                except IndexError:
                    f.write(f'{rank},{rnd},-1\n')
            f.write('\n')
    # output sync ratio in each round
    with open(f'results/mpi-ratio-{system_count}.csv', 'wt') as f:
        f.write('round,ratio\n')
        for rnd in range(len(t_exec_list[0])):
            f.write(f'{rnd},{t_sync_list[rank][rnd] / (t_sync_list[rank][rnd] + t_exec_list[rank][rnd])}\n')
    return [t_msg, t_sync, t_exec]


@ResultCallback.make(['msg', 'sync', 'exec', 'sorting', 'process', 'slowdown'])
def get_mtp_time() -> list:
    t_sync = 0
    t_exec = 0
    t_sync_list = []
    t_exec_list = []
    system_count = 0
    # calculate average time
    for filename in os.listdir('results'):
        if filename.startswith('MT-'):
            t_sync_round_list = []
            t_exec_round_list = []
            system_count += 1
            fullname = f'results/{filename}'
            with open(fullname, 'rt') as f:
                f.readline()
                line = f.readline().split(',')
                t_sync += int(line[-2])
                t_exec += int(line[-1])
                for line in f.readlines():
                    line = line.split(',')
                    t_sync_round_list.append(int(line[-2]))
                    t_exec_round_list.append(int(line[-1]))
            t_sync_list.append(t_sync_round_list)
            t_exec_list.append(t_exec_round_list)
            os.unlink(fullname)
    # get total time
    with open('results/MT.csv') as f:
        line = f.readlines()[-1].split(',')
        t_msg = int(line[-4])
        t_sorting = int(line[-3])
        t_process = int(line[-2])
        slowdown = float(line[-1])
        os.unlink('results/MT.csv')
    t_msg /= system_count * 1e9
    t_sync /= system_count * 1e9
    t_exec /= system_count * 1e9
    t_sorting /= 1e9
    t_process /= 1e9
    # output exec time in each round
    with open(f'results/mtp-exec-{system_count}.csv', 'wt') as f:
        f.write('rank,round,exec\n')
        for rank in range(system_count):
            for rnd in range(101):
                try:
                    f.write(f'{rank},{rnd},{sum(t_sync_list[rank][rnd * 100:rnd * 100 + 100])}\n')
                except IndexError:
                    f.write(f'{rank},{rnd},-1\n')
            f.write('\n')
    # output sync ratio in each round
    with open(f'results/mtp-ratio-{system_count}.csv', 'wt') as f:
        f.write('round,ratio\n')
        for rnd in range(len(t_exec_list[0])):
            f.write(f'{rnd},{t_sync_list[rank][rnd] / (t_sync_list[rank][rnd] + t_exec_list[rank][rnd])}\n')
    return [t_msg, t_sync, t_exec, t_sorting, t_process, slowdown]


@ResultCallback.make(['miss'])
def get_cache_miss() -> list:
    cache_miss = -1
    with open('results/perf.txt') as f:
        for line in f.readlines():
            if 'cache-misses' in line:
                cache_miss = int(line.strip().removesuffix('cache-misses').strip().replace(',', ''))
    os.unlink('results/perf.txt')
    return [cache_miss]


# --- experiment parameters ---


if __name__ == '__main__':
    if len(argv) != 2:
        print('Usage:')
        print('./exp.py init')
        print('./exp.py [experiment name]')
        exit(0)

    if argv[1] == 'init':
        subprocess.run('sudo apt update', shell=True)
        subprocess.run('sudo apt install build-essential cmake git openmpi-bin openmpi-doc libopenmpi-dev linux-tools-generic ninja-build -y', shell=True)
        subprocess.run('sudo sysctl -w kernel.perf_event_paranoid=-1', shell=True)
        subprocess.run('echo off | sudo tee /sys/devices/system/cpu/smt/control', shell=True)

    # Fig 1
    elif argv[1] == 'fat-tree':
        e = Experiment(argv[1])
        e.run('fat-tree', ['barrier', 'nullmsg', 'mtp'],
              k=8,
              cluster=[12, 24],
              delay=3000,
              bandwidth='100Gbps',
              flow=False,
              incast=1,
              victim=lambda args: '-'.join([str(i) for i in range(int(args['k']) ** 2 // 4)]),
              time=0.1,
              interval=0.01,
              flowmon=True,
              core=lambda args: args['cluster'])

    # Fig 1
    elif argv[1] == 'fat-tree-hybrid':
        e = Experiment(argv[1])
        e.run('fat-tree', ['barrier', 'nullmsg', 'hybrid'],
              k=8,
              cluster=[48, 72, 96, 120, 144],
              delay=3000,
              bandwidth='100Gbps',
              flow=False,
              incast=1,
              victim=lambda args: '-'.join([str(i) for i in range(int(args['k']) ** 2 // 4)]),
              time=0.1,
              interval=0.01,
              flowmon=True,
              core=lambda args: args['cluster'])

    # Fig 1
    elif argv[1] == 'fat-tree-default':
        e = Experiment(argv[1])
        e.run('fat-tree', 'default',
              k=8,
              cluster=[12, 24, 48, 72, 96, 120, 144],
              delay=3000,
              bandwidth='100Gbps',
              flow=False,
              incast=1,
              victim=lambda args: '-'.join([str(i) for i in range(int(args['k']) ** 2 // 4)]),
              time=0.1,
              interval=0.01,
              flowmon=True,
              core=1)

    # Fig 5a
    elif argv[1] == 'mpi-time-incast':
        e = Experiment(argv[1])
        e.run('fat-tree', ['nullmsg', 'barrier'],
              branch='unison-evaluations-for-mpi',
              callback=get_mpi_time,
              k=8,
              delay=3000,
              bandwidth='100Gbps',
              flow=False,
              incast=[0, 0.2, 0.4, 0.6, 0.8, 1],
              victim=lambda args: '-'.join([str(i) for i in range(int(args['k']) ** 2 // 4)]),
              time=0.1,
              interval=0.01,
              flowmon=True,
              core=8)

    # Fig 5b
    elif argv[1] == 'mpi-sync':
        e = Experiment(argv[1])
        e.run('fat-tree', 'barrier',
              branch='unison-evaluations-for-mpi',
              callback=get_mpi_time,
              k=8,
              delay=3000,
              bandwidth='100Gbps',
              flow=False,
              time=0.1,
              flowmon=True,
              core=8)

    # Fig 5c
    elif argv[1] == 'mpi-sync-delay':
        e = Experiment(argv[1])
        e.run('fat-tree', ['barrier', 'nullmsg'],
              branch='unison-evaluations-for-mpi',
              callback=get_mpi_time,
              k=8,
              delay=[3000000, 300000, 30000, 3000, 300],
              bandwidth='10Gbps',
              flow=False,
              time=0.1,
              interval=0.01,
              flowmon=True,
              core=8)

    # Fig 5d
    elif argv[1] == 'mpi-sync-bandwidth':
        e = Experiment(argv[1])
        e.run('fat-tree', ['barrier', 'nullmsg'],
              branch='unison-evaluations-for-mpi',
              callback=get_mpi_time,
              k=8,
              delay=30000,
              bandwidth=['2Gbps', '4Gbps', '6Gbps', '8Gbps', '10Gbps'],
              flow=False,
              load=lambda args: 1 / int(args['bandwidth'].removesuffix('Gbps')),
              time=0.1,
              interval=0.01,
              flowmon=True,
              core=8)

    # Fig 8
    elif argv[1] == 'dqn':
        e = Experiment(argv[1])
        e.run('dqn', ['barrier', 'nullmsg', 'mtp', 'default'],
              k=[4, 8],
              cluster=[4, 8],
              delay=500000,
              bandwidth='100Mbps',
              tcp='ns3::TcpCubic',
              load=0.7,
              time=20,
              interval=1,
              flowmon=True,
              core=16)

    # Fig 12
    elif argv[1] == 'mimic':
        e = Experiment(argv[1])
        e.run('fat-tree', ['barrier', 'nullmsg', 'mtp', 'default'],
              k=4,
              cluster=4,
              delay=500000,
              bandwidth='100Mbps',
              tcp='ns3::TcpCubic',
              load=0.7,
              flowmon=True,
              core=4)
        e.run('fat-tree', ['default'],
              k=4,
              cluster=2,
              delay=500000,
              bandwidth='100Mbps',
              tcp='ns3::TcpCubic',
              load=0.7,
              flowmon=True,
              core=1)

    # Fig 9
    elif argv[1] == 'flexible':
        e = Experiment(argv[1])
        e.run('fat-tree', 'mtp',
              k=8,
              delay=3000,
              bandwidth='100Gbps',
              core=[24, 20, 16, 12, 8, 4, 2])

    # Fig 9
    elif argv[1] == 'flexible-old':
        e = Experiment(argv[1])
        e.run('fat-tree', ['barrier', 'nullmsg'],
              k=8,
              delay=3000,
              bandwidth='100Gbps',
              core=[8, 4, 2])

    # Fig 9
    elif argv[1] == 'flexible-default':
        e = Experiment(argv[1])
        e.run('fat-tree', 'default',
              k=8,
              delay=3000,
              bandwidth='100Gbps')

    # Fig 10a
    elif argv[1] == 'mtp-sync-incast':
        e = Experiment(argv[1])
        e.run('fat-tree', 'mtp',
              branch='unison-evaluations-for-mtp',
              callback=get_mtp_time,
              k=8,
              delay=3000,
              bandwidth='100Gbps',
              flow=False,
              incast=[0, 0.2, 0.4, 0.6, 0.8, 1],
              victim=lambda args: '-'.join([str(i) for i in range(int(args['k']) ** 2 // 4)]),
              time=0.1,
              interval=0.01,
              flowmon=True,
              core=8)

    # Fig 10b
    elif argv[1] == 'mtp-sync':
        e = Experiment(argv[1])
        e.run('fat-tree', 'mtp',
              branch='unison-evaluations-for-mtp',
              callback=get_mtp_time,
              k=8,
              delay=3000,
              bandwidth='100Gbps',
              flow=False,
              time=0.1,
              flowmon=True,
              core=8)

    # Fig 11a
    elif argv[1] == 'torus-hybrid':
        e = Experiment(argv[1])
        e.run('torus', ['barrier', 'nullmsg', 'hybrid'],
              row=48,
              col=48,
              delay=30000,
              bandwidth='10Gbps',
              incast=0.5,
              time=1,
              core=[144, 96, 72, 48])

    # Fig 11a
    elif argv[1] == 'torus':
        e = Experiment(argv[1])
        e.run('torus', ['barrier', 'nullmsg', 'mtp'],
              row=48,
              col=48,
              delay=30000,
              bandwidth='10Gbps',
              incast=0.5,
              time=1,
              core=[24, 12, 6])
    
    # Fig 11a
    elif argv[1] == 'torus-default':
        e = Experiment(argv[1])
        e.run('torus', 'default',
              row=48,
              col=48,
              delay=30000,
              bandwidth='10Gbps',
              incast=0.5,
              time=1,
              core=1)

    # Fig 11b
    elif argv[1] == 'bcube':
        e = Experiment(argv[1])
        e.run('bcube', 'mtp',
              n=8,
              delay=3000,
              bandwidth='10Gbps',
              cdf=['scratch/cdf/google-rpc.txt', 'scratch/cdf/web-search.txt'],
              incast=0.5,
              victim=lambda args: '-'.join([str(i) for i in range(int(args['n']))]),
              time=0.1,
              interval=0.01,
              core=16)

    # Fig 11b
    elif argv[1] == 'bcube-old':
        e = Experiment(argv[1])
        e.run('bcube', ['barrier', 'nullmsg', 'mtp', 'default'],
              n=8,
              delay=3000,
              bandwidth='10Gbps',
              cdf=['scratch/cdf/google-rpc.txt', 'scratch/cdf/web-search.txt'],
              incast=0.5,
              victim=lambda args: '-'.join([str(i) for i in range(int(args['n']))]),
              time=0.1,
              interval=0.01,
              core=8)

    # Fig 11c
    elif argv[1] == 'wan':
        e = Experiment(argv[1])
        e.run('wan', ['mtp', 'default'],
              topo=['scratch/topos/geant.graphml', 'scratch/topos/chinanet.graphml'],
              delay=5000000,
              bandwidth='10Gbps',
              ecn=False,
              rip=True,
              tcp='ns3::TcpBbr',
              load=0.5,
              time=10,
              interval=1,
              core=16)

    # DCTCP reproduce
    elif argv[1] == 'reproduction':
        e = Experiment(argv[1])
        e.run('dctcp-example', ['default', 'mtp'],
              example=True)

    # Fig 13
    elif argv[1] == 'deterministic':
        e = Experiment(argv[1])
        e.run('fat-tree', ['barrier', 'nullmsg', 'mtp'],
              k=8,
              delay=500000,
              bandwidth=['100Mbps', '1Gbps', '10Gbps'],
              tcp='ns3::TcpCubic',
              load=0.7,
              flowmon=True,
              core=[8] * 10)

    # Fig 14a
    elif argv[1] == 'partition-cache':
        e = Experiment(argv[1])
        e.run('torus', 'mtp',
              template_cmd='perf stat -e cache-misses -o results/perf.txt',
              callback=get_cache_miss,
              row=12,
              col=12,
              delay=30000,
              incast=0.5,
              system=[144, 72, 48, 36, 24, 18, 12, 6, 4, 3, 2, 1],
              time=1,
              interval=0.1,
              core=1)

    # Fig 14b
    elif argv[1] == 'scheduling-metrics':
        e = Experiment(argv[1])
        e.run('fat-tree', 'mtp',
              branch='unison-evaluations-for-mtp',
              callback=get_mtp_time,
              k=8,
              sort=['None', 'ByExecutionTime', 'ByPendingEventCount'],
              core=[4, 8, 12, 16])

    # Fig 14c
    elif argv[1] == 'scheduling-period':
        e = Experiment(argv[1])
        e.run('fat-tree', 'mtp',
              branch='unison-evaluations-for-mtp',
              callback=get_mtp_time,
              k=8,
              period=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
              core=16)

    else:
        print('No such experiment!')
