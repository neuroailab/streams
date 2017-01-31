from __future__ import division, print_function, absolute_import
import sys, subprocess, time, glob, os, inspect, random, string

import dill
import tqdm
# from joblib import Parallel, delayed
# from multiprocessing import Pool


class Parallel(object):

    def __init__(self, func, n_jobs=None, n_iter=1, backend='sbatch', timer=False):
        self.func = func
        if n_jobs is None:
            self.n_jobs = n_iter
        else:
            self.n_jobs = min(n_jobs, n_iter)
        self.n_iter = n_iter
        self.backend = backend
        self.timer = timer

        template = 'parallel_' + self.pid + '_{}'
        self.save_path = os.path.join(os.getcwd(), template)

        frame = inspect.getouterframes(inspect.currentframe())[1][0]
        self._caller = inspect.getmodule(frame)
        self._caller_path = os.path.dirname(os.path.abspath(self._caller.__file__))

    def __call__(self, *args, **kwargs):
        if self.n_jobs == 1:  # run normally
            res = [self.func(i, *args, **kwargs) for i in tqdm.trange(self.n_iter)]
        else:
            if self.backend == 'sbatch':
                res = self._sbatch_run(*args, **kwargs)
            elif self.backend == 'pickle':
                res = self._run_from_pickle(*args, **kwargs)
            else:
                raise ValueError
        return res

    @property
    def pid(self):
        if not hasattr(self, '_pid'):
            if not hasattr(self, 'save_path'):
                self.save_path = os.getcwd()
            pid = self.id_generator(n=6)
            filelist = glob.glob(os.path.join(os.path.dirname(self.save_path), '*'))
            while any([pid in os.path.basename(f) for f in filelist]):
                pid = self.id_generator(m=6)
            self._pid = pid
        return self._pid

    def id_generator(self, n=6, chars=string.ascii_lowercase):
        return ''.join(random.SystemRandom().choice(chars) for _ in range(n))

    def _func_writer(self, iterno, *args, **kwargs):
        res = self.func(iterno, *args, **kwargs)
        tempf = self.save_path.format('output_{}.pkl'.format(iterno))
        dill.dump(res, open(tempf, 'wb'))

    def gen_sbatch_array(self, array=(0,100)):
        tempf = self.save_path.format('slurm_%a.out')
        callable_path = self.save_path.format('script.py')
        script = ('#!/bin/bash\n'
                  '#SBATCH --array={}-{}\n'
                  '#SBATCH --time=01:00:00\n'
                  '#SBATCH --ntasks=1\n'
                  '#SBATCH --output="{}"\n'
                  'python "{}" $SLURM_ARRAY_TASK_ID'.format(
                      array[0], array[1]-1, tempf, callable_path)
                  )
        with open(self.save_path.format('sbatch.sh'), 'wb') as f:
            f.write(script)

    def gen_sbatch_python(self, *args, **kwargs):
        tempf = self.save_path.format('vars.pkl')
        dill.dump([self._caller, self._func_writer, args, kwargs], open(tempf, 'wb'))
        # add all modules
        global_vars = ''
        for global_var in dir(self._caller):
            if not global_var[:2] == '__' and not global_var[-2:] == '__':  # no built-in
                global_vars += '    {} = mod.{}\n'.format(global_var, global_var)

        script = ('import sys\n'
                  'def run(iterno):\n'
                  '    import sys, dill\n'
                  '    sys.path.insert(0, "{}")\n'
                  '    mod, func, args, kwargs = dill.load(open("{}"))\n{}'
                  '    func(iterno, *args, **kwargs)\n'
                  'if __name__ == "__main__":\n'
                  '    run(int(sys.argv[1]))'.format(self._caller_path, tempf, global_vars)
                  )
        with open(self.save_path.format('script.py'), 'wb') as f:
            f.write(script)

    def _run_from_pickle(self, *args, **kwargs):
        self.gen_sbatch_python(*args, **kwargs)
        for i in range(self.n_iter):
            subprocess.check_call('python {} {}'.format(sbatch_python.name, i).split())
        os.remove(func_pkl.name)
        os.remove(sbatch_python.name)
        p = os.path.basename(sbatch_python.name).split('.')[0]
        return self._combine_results('parallel_output_{}_*_{}.pkl'.format(p))

    def _sbatch_run(self, *args, **kwargs):
        self.gen_sbatch_python(*args, **kwargs)
        for batch_no in range((self.n_jobs - 1) // self.n_iter + 1):
            array = (batch_no * self.n_jobs, (batch_no+1) * self.n_jobs)
            self.gen_sbatch_array(array=array)

            # run the script and wait for it to complete
            out = subprocess.check_output(['sbatch', self.save_path.format('sbatch.sh')])
            out_str = 'Submitted batch job '
            assert out[:len(out_str)] == out_str
            job_id = out.split('\n')[0][len(out_str):]
            while True:
                jobs = subprocess.check_output('squeue -o %M -j {}'.format(job_id).split())
                if len(jobs.split('\n')) > 2:  # still running
                    if self.timer:
                        t = jobs.split('\n')[-2]
                        print('\rJob {}: {}'.format(job_id, t), end='')
                        sys.stdout.flush()
                    time.sleep(1)
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


# def parallel(n_jobs=-1, n_parallel=10):
#     def _parallel(func):
#         def func_wrapper(*args, **kwargs):
#             p = Parallel(n_jobs=n_jobs)
#             import pdb; pdb.set_trace()
#             return p(delayed(func)(*args, **kwargs) for i in range(n_parallel))
#         return func_wrapper
#     return _parallel
