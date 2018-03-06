import sys, subprocess, time, glob, os, random, string, pickle
import inspect, argparse, threading, signal, ast

import tqdm
import dill
# from joblib import Parallel, delayed
import multiprocessing

import numpy as np
import pandas

COMPUTED = os.environ.get('COMPUTED', '')


class Parallel(object):

    def __init__(self, func, n_iter, backend=None, timer=False, *args, **kwargs):
        self.func = func
        self.n_iter = n_iter
        self.backend = backend
        self.timer = timer

        if backend is None:
            self.parallel = 'loop'
        elif backend == 'sbatch':
            self.parallel = 'sbatch'
        elif backend == 'sbatch_pickle':
            self.parallel = SBatchPickle(*args, **kwargs)
        elif backend == 'multiprocessing':
            self.parallel = MultiProcessing(*args, **kwargs)
        else:
            raise ValueError('backend "{}" not recognized'.format(backend))

    def __call__(self, *args, **kwargs):
        if self.backend is None:  # normal loop
            return [self.func(iterno, *args, **kwargs) for iterno in tqdm.trange(self.n_iter, disable=not self.timer)]

        elif self.backend == 'sbatch':
            iternos = os.environ['PARALLEL_IDX']
            iternos = iternos.split('_')
            iterno = int(iternos[0])
            if len(iternos) > 1:
                os.environ['PARALLEL_IDX'] = '_'.join(iternos[1:])
            return self.func(iterno, *args, **kwargs)

        else:
            return self.parallel(*args, **kwargs)


def prange(start=0, stop=None, step=1, backend=None, **tqdm_kwargs):
    try:
        iter(start)
    except TypeError:
        rng = np.arange(start, stop, step)
    else:
        rng = list(start)

    if backend is None:
        if os.uname()[1].startswith('node0'):
            backend = 'sbatch'

    if backend is None:
        return tqdm.tqdm(rng, **tqdm_kwargs)
    elif backend == 'sbatch':
        iternos = os.environ['PARALLEL_IDX']
        iternos = iternos.split('_')
        iterno = int(iternos[0])
        if len(iternos) > 1:
            os.environ['PARALLEL_IDX'] = '_'.join(iternos[1:])
        return [rng[iterno]]
    # elif backend == 'multiprocessing':
    #     # results = []
    #     # for batch_no in tqdm.trange((self.n_iter - 1) // self.n_jobs + 1):
    #     pool = multiprocessing.Pool(processes=len(rng))
    #     # array = range(batch_no * self.n_jobs, (batch_no+1) * self.n_jobs)
    #     if hasattr(pool, 'starmap'):
    #         out = pool.starmap(self.func, ([i, args, kwargs] for i in rng))
    #     else:
    #         func_args = ([self.func, i, args, kwargs] for i in rng)
    #         out = pool.map(func_star, func_args)
    #     pool.close()
    #     pool.join()
    #     # results.extend(out)
    #     return out
    else:
        raise ValueError('backend "{}" not recognized'.format(backend))


class ParallelBase(object):

    @property
    def pid(self):
        if not hasattr(self, '_pid'):
            pid = self.id_generator(n=6)
            filelist = glob.glob(os.path.join(self._save_path, '*'))
            while any([pid in os.path.basename(f) for f in filelist]):
                pid = self.id_generator(m=6)
            self._pid = pid
        return self._pid

    def id_generator(self, n=6, chars=string.ascii_lowercase):
        return ''.join(random.SystemRandom().choice(chars) for _ in range(n))


class SBatch(object):

    def __init__(self, module, func, output_path=None, output_name=None, timer=False, save_path=None, **func_kwargs):
        self.module = os.path.abspath(module)
        self.func = func
        self.func_kwargs = func_kwargs

        if output_path is None:
            rel_path = os.path.relpath(os.path.splitext(self.module)[0], os.environ['CODE'])
            output_path = os.path.join(COMPUTED, rel_path)
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        if output_name is None:
            output_name = self.func + '.pkl'
        self.output_file = os.path.join(output_path, output_name)

        self.timer = timer
        if save_path is None:
            self._save_path = '/om/user/qbilius/tmp'  # os.getcwd()
        else:
            self._save_path = save_path

        template = self.pid + '_{}'
        self.save_path = os.path.join(self._save_path, template)

        self._stop = threading.Event()
        signal.signal(signal.SIGINT, self._break)

    def __call__(self, n_iters):
        python_script_path = self.gen_python_script()
        try:
            iter(n_iters)
        except:
            self.n_iters = [n_iters]
        else:
            self.n_iters = n_iters
        array = (0, np.prod(self.n_iters))
        sbatch_path = self.gen_sbatch(python_script_path, array=array)
        out = subprocess.check_output(['sbatch', sbatch_path])
        out = out.decode('ascii')
        out_str = 'Submitted batch job '
        assert out[:len(out_str)] == out_str
        self.job_id = out.split('\n')[0][len(out_str):]
        self.sbatch_progress()

        if not self._stop.is_set():
            # check if output file was created; if not, there must have been an error
            for i in range(array[0], array[1]):
                outf = self.save_path.format('output_{}.pkl'.format(i))
                if not os.path.isfile(outf):
                    with open(self.save_path.format('slurm_{}.out'.format(i))) as f:
                        msg = f.read()
                    print()
                    print(msg)
                    self._cleanup()
                    sys.exit()
        else:
            print('cleaning up...')
            self._cleanup()
            sys.exit()

        results = self._combine_results()
        self._cleanup()
        pickle.dump(results, open(self.output_file, 'wb'))
        return results

    @property
    def pid(self):
        if not hasattr(self, '_pid'):
            pid = self.id_generator(n=6)
            filelist = glob.glob(os.path.join(self._save_path, '*'))
            while any([pid in os.path.basename(f) for f in filelist]):
                pid = self.id_generator(m=6)
            self._pid = pid
        return self._pid

    def id_generator(self, n=6, chars=string.ascii_lowercase):
        return ''.join(random.SystemRandom().choice(chars) for _ in range(n))

    def gen_python_script(self):
        mod_name = os.path.splitext(os.path.basename(self.module))[0]
        output_name = self.save_path.format('output_{}.pkl')
        script = ('import sys, os, imp, pickle',
                  'import numpy as np',
                  'sys.path.insert(0, "{}")',
                  'mod = imp.load_source("{}", "{}")',
                  'sh = [int(i) for i in os.environ["PARALLEL_SHAPE"].split("_")]',
                  'task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])',
                  'n_iters = np.prod(sh)',
                  'idx = np.nonzero(np.arange(n_iters).reshape(sh)==task_id)',
                  'idx = "_".join(str(v[0]) for v in idx)',
                  'os.environ["PARALLEL_IDX"] = idx',
                  'res = getattr(mod, "{}")({})',
                  'pickle.dump(res, open("{}".format(task_id), "wb"))'
                  )
        kwargs = []
        for k,v in self.func_kwargs.items():
            if isinstance(k, str):
                inp = '{}="{}"'.format(k, v)
            else:
                inp = '{}={}'.format(k, v)
            kwargs.append(inp)
        kwargs = ', '.join(kwargs)
        script = '\n'.join(script).format(os.getcwd(), mod_name, self.module, self.func, kwargs, output_name)
        script_path = self.save_path.format('script.py')
        with open(script_path, 'w') as f:
            f.write(script)
        return script_path

    def gen_sbatch(self, callable_path, array=(0,100)):
        slurm_out_file = self.save_path.format('slurm_%a.out')
        script = ('#!/bin/bash',
                  '#SBATCH --array={}-{}',
                  '#SBATCH --time=7-00:00:00',
                  '#SBATCH --ntasks=1',
                  '#SBATCH --cpus-per-task=1',
                  '#SBATCH --mem=100G',
                  '#SBATCH --output="{}"',
                  'export PARALLEL_SHAPE={}',
                  'python "{}" $SLURM_ARRAY_TASK_ID')
        shape = '_'.join([str(i) for i in self.n_iters])
        script = '\n'.join(script).format(array[0], array[1] - 1, slurm_out_file,
                                          shape, callable_path)
        sbatch_path = self.save_path.format('sbatch.sh')
        with open(sbatch_path, 'w') as f:
            f.write(script)
        return sbatch_path

    def sbatch_progress_orig(self, job_id):
        while not self._stop.is_set():
            try:
                jobs = subprocess.check_output('squeue -o %M -j {}'.format(job_id).split())
            except:
                print('Squeue error. Trying again in 10 sec...')
                self._stop.wait(10)  # queue busy, ask later
            else:
                jobs = jobs.decode('ascii')
                if not jobs.startswith('TIME'):
                    print('Unexpected squeue output. Trying again in 10 sec...')
                    self._stop.wait(10)  # queue busy, ask later
                elif len(jobs.split('\n')) > 2:  # still running
                    if self.timer:
                        t = jobs.split('\n')[-2]
                        print('\rJob {} (id: {}): {}'.format(job_id, self.pid, t), end='')
                        sys.stdout.flush()
                    self._stop.wait(10)
                else:
                    break
        if self.timer:
            print()

    def sbatch_progress(self, wait=10):
        if self.timer:
            os.system('setterm -cursor off')
        while not self._stop.is_set():
            try:
                cmd = 'sacct -j {} -o State -X'.format(self.job_id).split()
                jobs = subprocess.check_output(cmd)
            except:
                print('\rSqueue error. Trying again in {} sec...'.format(wait),
                      end='', flush=True)
                self._stop.wait(wait)  # queue busy, ask later
            else:
                jobs = jobs.decode('ascii')
                if not jobs.startswith('     State'):
                    print('\rUnexpected squeue output. Trying again in {} sec...'.format(wait),
                          end='', flush=True)
                    self._stop.wait(wait)  # queue busy, ask later
                else:
                    status = jobs.split('\n')[2:-1]
                    count = {}
                    for st in status:
                        st = st.strip('\r\n').lstrip().rstrip()
                        if st in count:
                            count[st] += 1
                        else:
                            count[st] = 1
                    status = ', '.join(['{}: {}'.format(k,v) for k,v in count.items()])
                    if self.timer:
                        print('\r' + ' ' * 79, end='')
                        print('\rJob {} (id: {}) -- {}'.format(self.job_id, self.pid, status),
                              end='', flush=True)
                    if 'COMPLETED' in count and len(count) == 1:
                        break
                    self._stop.wait(wait)

        if self.timer:
            os.system('setterm -cursor on')
            print()

    def _break(self, signum, frame):
        self._stop.set()
        subprocess.check_output(['scancel', self.job_id])

    def _cleanup(self):
        for fname in ['sbatch.sh', 'script.py']:
            try:
                os.remove(self.save_path.format(fname))
            except:
                pass
        for fname in ['slurm_{}.out', 'output_{}.pkl']:
            for i in range(np.prod(self.n_iters)):
                try:
                    os.remove(self.save_path.format(fname).format(i))
                except:
                    pass

    def _combine_results(self):
        results = []
        for i in range(np.prod(self.n_iters)):
            outf = self.save_path.format('output_{}.pkl'.format(i))
            res = pickle.load(open(outf, 'rb'))
            results.append(res)

        try:
            results = pandas.concat(results, ignore_index=True)
        except:
            pass
        return results


class SBatchPickle(ParallelBase):

    def __init__(self, func, n_jobs=None, n_iter=1, backend='sbatch', timer=False,
                 save_path=None):
        self.func = func
        if n_jobs is None:
            self.n_jobs = n_iter
        else:
            self.n_jobs = min(n_jobs, n_iter)
        self.n_iter = n_iter
        self.backend = backend
        self.timer = timer
        if save_path is None:
            self._save_path = '/om/user/qbilius/tmp'  # os.getcwd()
        else:
            self._save_path = save_path

        template = 'parallel_' + self.pid + '_{}'
        self.save_path = os.path.join(self._save_path, template)

        if self.backend == 'sbatch':
            frames = inspect.getouterframes(inspect.currentframe())
            self._callers = [inspect.getmodule(frame[0]) for frame in frames[::-1]]
            self._callers = [c for c in self._callers if c is not None]
            self._callers = self._callers[-1:]
            # self._caller = inspect.getmodule(frame)
            # self._caller_path = os.path.dirname(os.path.abspath(self._caller.__file__))
            # import pdb; pdb.set_trace()

    def __call__(self, *args, **kwargs):
        if self.n_jobs == 1:  # run normally
            res = [self.func(i, *args, **kwargs) for i in tqdm.trange(self.n_iter)]
        else:
            res = self._sbatch_run(*args, **kwargs)
        return res

    def _func_writer(self, iterno, *args, **kwargs):
        res = self.func(iterno, *args, **kwargs)
        tempf = self.save_path.format('output_{}.pkl'.format(iterno))
        dill.dump(res, open(tempf, 'wb'))

    def gen_sbatch_array(self, array=(0,100)):
        tempf = self.save_path.format('slurm_%a.out')
        callable_path = self.save_path.format('script.py')
        script = ('#!/bin/bash\n'
                  '#SBATCH --array={}-{}\n'
                  '#SBATCH --time=7-00:00:00\n'
                  '#SBATCH --ntasks=1\n'
                  '#SBATCH --output="{}"\n'
                  'python "{}" $SLURM_ARRAY_TASK_ID'.format(
                      array[0], array[1]-1, tempf, callable_path)
                  )
        with open(self.save_path.format('sbatch.sh'), 'wb') as f:
            f.write(script)

    def gen_sbatch_python(self, *args, **kwargs):
        tempf = self.save_path.format('vars.pkl')
        dill.dump([self._callers, self._func_writer, args, kwargs], open(tempf, 'wb'))

        script = ('import sys\n'
                  'def run(iterno):\n'
                  '    import sys, dill\n'
                  '    sys.path.insert(0, "{}")\n'.format(os.getcwd()))

        # add all modules
        for caller in self._callers:
            path = os.path.dirname(os.path.abspath(caller.__file__))
            script += '    sys.path.insert(0, "{}")\n'.format(path)

        script += '    mods, func, args, kwargs = dill.load(open("{}"))\n'.format(tempf)
        for i, caller in enumerate(self._callers):
            for global_var in dir(caller):
                if not global_var[:2] == '__' and not global_var[-2:] == '__':  # no built-in
                    script += '    {} = mods[{}].{}\n'.format(global_var, i, global_var)

        script += ('    func(iterno, *args, **kwargs)\n'
                   'if __name__ == "__main__":\n'
                   '    run(int(sys.argv[1]))')

        with open(self.save_path.format('script.py'), 'wb') as f:
            f.write(script)

    def _sbatch_run(self, *args, **kwargs):
        self.gen_sbatch_python(*args, **kwargs)
        for batch_no in range((self.n_iter - 1) // self.n_jobs + 1):
            array = (batch_no * self.n_jobs, (batch_no+1) * self.n_jobs)
            self.gen_sbatch_array(array=array)

            # run the script and wait for it to complete
            out = subprocess.check_output(['sbatch', self.save_path.format('sbatch.sh')])
            out_str = 'Submitted batch job '
            assert out[:len(out_str)] == out_str
            job_id = out.split('\n')[0][len(out_str):]
            while True:
                try:
                    jobs = subprocess.check_output('squeue -o %M -j {}'.format(job_id).split())
                except:
                    print('Squeue error. Trying again in 10 sec...')
                    time.sleep(10)  # queue busy, ask later
                else:
                    if not jobs.startswith('TIME'):
                        print('Unexpected squeue output. Trying again in 10 sec...')
                        time.sleep(10)  # queue busy, ask later
                    elif len(jobs.split('\n')) > 2:  # still running
                        if self.timer:
                            t = jobs.split('\n')[-2]
                            print('\rJob {}: {}'.format(job_id, t), end='')
                            sys.stdout.flush()
                        time.sleep(10)
                    else:
                        break

            # check if output file was created; if not, there must have been an error
            for i in range(array[0], array[1]):
                outf = self.save_path.format('output_{}.pkl'.format(i))
                if not os.path.isfile(outf):
                    with open(self.save_path.format('slurm_{}.out'.format(i))) as f:
                        msg = f.read()
                    print(msg)
                    raise Exception('Output file {} not found. See error log above.'.format(outf))

        results = self._combine_results()
        self._cleanup()
        return results

    def _cleanup(self):
        for fname in ['sbatch.sh', 'script.py', 'vars.pkl']:
            os.remove(self.save_path.format(fname))
        for fname in ['slurm_{}.out', 'output_{}.pkl']:
            for i in range(self.n_iter):
                os.remove(self.save_path.format(fname).format(i))

    def _combine_results(self):
        results = []
        for i in range(self.n_iter):
            outf = self.save_path.format('output_{}.pkl'.format(i))
            res = dill.load(open(outf))
            results.append(res)
        return results


class MultiProcessing(ParallelBase):

    def __init__(self, func, n_jobs=None, n_iter=1, backend='sbatch', timer=False,
                 save_path=None):
        self.func = func
        if n_jobs is None:
            self.n_jobs = n_iter
        else:
            self.n_jobs = min(n_jobs, n_iter)
        self.n_iter = n_iter
        self.backend = backend
        self.timer = timer
        if save_path is None:
            self._save_path = '/om/user/qbilius/tmp'  # os.getcwd()
        else:
            self._save_path = save_path

        template = 'parallel_' + self.pid + '_{}'
        self.save_path = os.path.join(self._save_path, template)

    def __call__(self, *args, **kwargs):
        if self.n_jobs == 1:  # run normally
            res = [self.func(i, *args, **kwargs) for i in tqdm.trange(self.n_iter)]
        else:
            res = self._sbatch_run(*args, **kwargs)
        return res

    def _multiproc_run(self, *args, **kwargs):
        results = []
        for batch_no in tqdm.trange((self.n_iter - 1) // self.n_jobs + 1):
            pool = multiprocessing.Pool(processes=self.n_jobs)
            array = range(batch_no * self.n_jobs, (batch_no+1) * self.n_jobs)
            if hasattr(pool, 'starmap'):
                out = pool.starmap(self.func, ([i, args, kwargs] for i in array))
            else:
                func_args = ([self.func, i, args, kwargs] for i in array)
                out = pool.map(func_star, func_args)
            pool.close()
            pool.join()
            results.extend(out)
        return results


def func_star(args):
    func, iterno, args, kwargs = args
    return func(iterno, *args, **kwargs)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('module', help='path to the Python script you want to run')
    parser.add_argument('func', help='function to call')
    parser.add_argument('n_iters', help='number of iterations')
    parser.add_argument('-o', '--output_name', default=None, help='combined file name')
    parser.add_argument('--output_path', '--output_path', default=None, help='where to save the combined file')
    # parser.add_argument('-p', '--output_prefix', default='', help='prefix to the output path')
    parser.add_argument('--timer', default=True, action='store_true', help='whether to show a timer')
    parser.add_argument('--save_path', default=None, help='temporary place for storing intermediate results')

    args, func_args = parser.parse_known_args()
    func_kwargs = {k.strip('-'):v for k,v in zip(*[iter(func_args)] * 2)}
    for k, v in func_kwargs.items():
        try:
            func_kwargs[k] = ast.literal_eval(v)
        except:
            pass
    kwargs = {k:v for k,v in args.__dict__.items() if k != 'n_iters'}
    kwargs.update(func_kwargs)
    SBatch(**kwargs)(eval(args.__dict__['n_iters']))


if __name__ == '__main__':
    run()