from __future__ import division, print_function, absolute_import
import sys, subprocess, tempfile, time, glob, os, inspect, importlib

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
        self.save_path = os.getcwd()

        frame = inspect.getouterframes(inspect.currentframe())[1][0]
        self._caller = inspect.getmodule(frame)
        self._caller_path = os.path.dirname(os.path.abspath(self._caller.__file__))

    def __call__(self, *args, **kwargs):
        if self.n_jobs == 1:  # run normally
            res = [self.func(*args, **kwargs) for i in tqdm.trange(self.n_iter)]
        else:
            if self.backend == 'sbatch':
                res = self._sbatch_run(*args, **kwargs)
            elif self.backend == 'pickle':
                res = self._run_from_pickle(*args, **kwargs)
            else:
                raise ValueError
        return res

    def _func_writer(self, iterno, *args, **kwargs):
        res = self.func(iterno, *args, **kwargs)
        name = os.path.splitext(os.path.basename(sys.argv[0]))[0].split('_')[-1]
        tempf = tempfile.NamedTemporaryFile(
                                    prefix='parallel_output_{}_'.format(name),
                                    suffix='_{}.pkl'.format(iterno),
                                    delete=False, dir=self.save_path)
        dill.dump(res, tempf)
        tempf.close()

    def gen_sbatch_array(self, callable_path, array=(0,100)):
        sbatch_script = tempfile.NamedTemporaryFile(prefix='parallel_sbatch_', suffix='.sh',
                                                    delete=False, dir=self.save_path)
        script = ('#!/bin/bash\n'
                  '#SBATCH --array={}-{}\n'
                  '#SBATCH --time=01:00:00\n'
                  '#SBATCH --ntasks=1\n'
                  '#SBATCH --output="parallel_slurm_%A-%a.out"\n'
                  'python "{}" $SLURM_ARRAY_TASK_ID'.format(array[0], array[1]-1, callable_path)
                  )
        sbatch_script.write(script)
        sbatch_script.close()
        return sbatch_script

    def gen_sbatch_python(self, *args, **kwargs):
        func_pkl = tempfile.NamedTemporaryFile(prefix='parallel_vars_', suffix='.pkl',
                                               delete=False, dir=self.save_path)
        dill.dump([self._caller, self._func_writer, args, kwargs], func_pkl)
        func_pkl.close()

        sbatch_python = tempfile.NamedTemporaryFile(prefix='parallel_script_', suffix='.py',
                                                    delete=False, dir=self.save_path)

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
                  '    run(int(sys.argv[1]))'.format(self._caller_path, func_pkl.name, global_vars)
                  )
        sbatch_python.write(script)
        sbatch_python.close()
        return func_pkl, sbatch_python

    def _run_from_pickle(self, *args, **kwargs):
        func_pkl, sbatch_python = self.gen_sbatch_python(*args, **kwargs)
        for i in range(self.n_iter):
            subprocess.check_call('python {} {}'.format(sbatch_python.name, i).split())
        os.remove(func_pkl.name)
        os.remove(sbatch_python.name)
        p = os.path.basename(sbatch_python.name).split('.')[0]
        return self._combine_results('parallel_output_{}_*_{}.pkl'.format(p))

    def _sbatch_run(self, *args, **kwargs):
        func_pkl, sbatch_python = self.gen_sbatch_python(*args, **kwargs)

        sbatch_script_names = []
        for batch_no in range((self.n_jobs - 1) // self.n_iter + 1):
            array = (batch_no * self.n_jobs, (batch_no+1) * self.n_jobs)
            sbatch_script = self.gen_sbatch_array(sbatch_python.name, array=array)
            sbatch_script_names.append(sbatch_script.name)
            out = subprocess.check_output(['sbatch', sbatch_script.name])
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

        p = os.path.splitext(os.path.basename(sbatch_python.name))[0].split('_')[-1]
        output_format = 'parallel_output_{}'.format(p) + '_*_{}.pkl'
        res_file = glob.glob(os.path.join(self.save_path, output_format.format(0)))
        assert len(res_file) == 1
        results = self._combine_results(output_format)

        for name in sbatch_script_names:
            os.remove(name)
        for i in range(self.n_iter):
            f = os.path.join(self.save_path, 'parallel_slurm_{}-{}.out'.format(job_id, i))
            os.remove(f)
        os.remove(func_pkl.name)
        os.remove(sbatch_python.name)
        return results

    def _combine_results(self, output_format):
        results = []
        for i in range(self.n_iter):
            res_file = glob.glob(os.path.join(self.save_path, output_format.format(i)))
            assert len(res_file) == 1
            res = dill.load(open(res_file[0]))
            results.append(res)
            os.remove(res_file[0])
        return results


# def parallel(n_jobs=-1, n_parallel=10):
#     def _parallel(func):
#         def func_wrapper(*args, **kwargs):
#             p = Parallel(n_jobs=n_jobs)
#             import pdb; pdb.set_trace()
#             return p(delayed(func)(*args, **kwargs) for i in range(n_parallel))
#         return func_wrapper
#     return _parallel
