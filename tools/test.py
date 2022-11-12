# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import importlib
import os
import os.path as osp
import pickle
import trace
from copy import deepcopy
from typing import List

import pandas as pd
from mmengine import DictAction

from mmdeploy.apis import build_task_processor
from mmdeploy.utils.config_utils import load_config
from mmdeploy.utils.timer import TimeCounter


class MyTrace(object):
    """trace what you want Two files will be output finally:

    '.pkl' file:     A python module that enabless objects to be serialized to
    files on disk.     Record all debugging information. '.csv' file: Record
    all information of function state transition, include the annotation and
    parameter.
    """

    def __init__(self, ignoremods: List[str], ignoredirs: List[str],
                 filtermods: List[str], renamemods: List[str], filename: str,
                 funcname: str, if_annotation: bool):
        """
            Args:
            ignoremods:
                a list of the names of modules to ignore.
            ignoredirs:
                a list of the names of directories to ignore
                all of the (recursive) contents of
            filtermods:
                Select the module to be filtered.
            renamemods:
                Select the module whose name needs to be replaced.
                (If you don't do this, the file name will be very long unless you want to debug the code)
            filename:
                Export address of the 'pkl' file.
            funcname:
                The program entrance that you want to track.
            if_annotation:
                Decide whether to start annotation printing, If false,
                Corresponding function annotation will not be obtained and printed into csv file.
        """
        self.outfile = filename
        self.func = funcname
        self.filtermods = filtermods
        self.renamemods = renamemods
        self.annotation = if_annotation
        self._init_outfile()
        self.trace = trace.Trace(
            ignoremods, ignoredirs, countfuncs=True, outfile=self.outfile)

    def _init_outfile(self):
        if not os.path.exists('./trace_file'):
            os.mkdir('trace_file')
        self.outfile = './trace_file' + '/' + self.outfile
        if not os.path.exists(self.outfile):
            with open(self.outfile, 'wb'):
                pass
        else:
            os.remove(self.outfile)
            with open(self.outfile, 'wb'):
                pass

    def _get_trace_pkl(self):
        self.trace.run(self.func)
        r = self.trace.results()
        r.write_results(show_missing=False, coverdir=None)

    def _result_loader(self):
        pkl_file = open(self.outfile, 'rb')
        pkl_data = pickle.load(pkl_file)
        pkl_file.close()
        return pkl_data

    def _result_clean(self, data_list):
        """maybe repeatedly delete 'j' will throw an exception."""
        for i in data_list.copy():
            for j in self.filtermods:
                try:
                    if j in i[0]:
                        data_list.remove(i)
                except:
                    pass
        return data_list

    def _result_rename(self, data_list):
        for n, i in enumerate(data_list):
            for j in self.renamemods:
                if j in i[0]:
                    data_list[n][0] = i[0].split(j + '/')[-1]
        return data_list

    @classmethod
    def debug_print(filename='trace_file/trace_result.pkl'):
        if os.path.exists(filename):
            pkl_file = open(filename, 'rb')
            pkl_data = pickle.load(pkl_file)
            pkl_file.close()
        else:
            raise (IOError, f'you should confirm {filename} is existed')

    def get_pkl_list(self) -> list:
        self._get_trace_pkl()
        pkl_result = self._result_loader()
        pkl_list = [list(i[0]) for i in pkl_result[1].items()]
        pkl_list = self._result_clean(pkl_list)
        pkl_list = self._result_rename(pkl_list)
        print('get_pkl_list success!')
        return pkl_list

    def get_result_csv(self, data_list: list):
        df = pd.DataFrame(columns=[
            'path', 'file_name', 'func_name', 'annotation', 'parameter'
        ])
        for i in data_list:
            df.loc[df.shape[0]] = dict(zip(df.columns, i))

        if self.annotation:
            self.get_annotation(df)
        df.to_csv(self.outfile.split('.pkl')[0] + '.csv')

    @staticmethod
    def run():
        pkl_list = my_trace.get_pkl_list()
        pkl_list = custom_filter(pkl_list)
        print(pkl_list)
        # output the result
        my_trace.get_result_csv(pkl_list)

    def get_annotation(self, df):
        df['annotation'] = 'None'
        df['parameter'] = 'None'
        for i in range(len(df)):
            file_name = df.iloc[i, 0]
            func_name = df.iloc[i, 2]
            import_from_file = 0
            if type(file_name) == str:
                file_import_name = file_name.split('.py')[0]
                file_import_name = file_import_name.replace('/', '.')

            if 'mmengine' in file_import_name and type(
                    file_import_name) == str:
                import_from_file = 'mmengine' + file_import_name.split(
                    'mmengine')[-1]

                import_module = importlib.import_module(import_from_file)
                assert type(import_module) == type(importlib)
                if func_name.split('.')[0] in dir(import_module) and type(
                        import_from_file) == str:
                    function = eval(import_from_file + '.' + func_name)
                    assert callable(function)

                    if hasattr(function,
                               '__doc__') and (function.__doc__ != ''):
                        df['annotation'][i] = function.__doc__
                    if hasattr(function, '__code__'):
                        try:
                            df['parameter'][i] = function.__code__.co_varnames
                        except:
                            pass

            _file_import_name = file_import_name
            if 'mmdet' in _file_import_name.split('.') and type(
                    file_import_name) == str:
                import_from_file = 'mmdet' + file_import_name.split(
                    'mmdet')[-1]
                import_module = importlib.import_module(import_from_file)
                assert type(import_module) == type(importlib)
                if func_name.split('.')[0] in dir(import_module) and type(
                        import_from_file) == str:
                    try:
                        function = eval(import_from_file + '.' + func_name)
                    except:
                        function = eval(import_from_file)

                    assert callable(function)
                    if hasattr(function,
                               '__doc__') and (function.__doc__ != ''):
                        df['annotation'][i] = function.__doc__
                    if hasattr(function, '__code__'):
                        try:
                            df['parameter'][i] = function.__code__.co_varnames
                        except:
                            pass


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDeploy test (and eval) a backend.')
    parser.add_argument('deploy_cfg', help='Deploy config path')
    parser.add_argument('model_cfg', help='Model config path')
    parser.add_argument(
        '--model', type=str, nargs='+', help='Input model files.')
    parser.add_argument(
        '--device', help='device used for conversion', default='cpu')
    parser.add_argument(
        '--work-dir',
        default='./work_dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='visualize per interval samples.')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=2,
        help='display time of every window. (second)')
    parser.add_argument(
        '--log2file',
        type=str,
        help='log evaluation results and speed to file',
        default=None)
    parser.add_argument(
        '--speed-test', action='store_true', help='activate speed test')
    parser.add_argument(
        '--warmup',
        type=int,
        help='warmup before counting inference elapse, require setting '
        'speed-test first',
        default=10)
    parser.add_argument(
        '--log-interval',
        type=int,
        help='the interval between each log, require setting '
        'speed-test first',
        default=100)
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='the batch size for test, would override `samples_per_gpu`'
        'in  data config.')
    parser.add_argument(
        '--uri',
        action='store_true',
        default='192.168.1.1:60000',
        help='Remote ipv4:port or ipv6:port for inference on edge device.')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    deploy_cfg_path = args.deploy_cfg
    model_cfg_path = args.model_cfg

    # load deploy_cfg
    deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        work_dir = args.work_dir
    elif model_cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])

    # merge options for model cfg
    if args.cfg_options is not None:
        model_cfg.merge_from_dict(args.cfg_options)

    task_processor = build_task_processor(model_cfg, deploy_cfg, args.device)

    # prepare the dataset loader
    test_dataloader = deepcopy(model_cfg['test_dataloader'])
    if type(test_dataloader) == list:
        dataset = []
        for loader in test_dataloader:
            ds = task_processor.build_dataset(loader['dataset'])
            dataset.append(ds)
            loader['dataset'] = ds
            loader['batch_size'] = args.batch_size
            loader = task_processor.build_dataloader(loader)
        dataloader = test_dataloader
    else:
        test_dataloader['batch_size'] = args.batch_size
        dataset = task_processor.build_dataset(test_dataloader['dataset'])
        test_dataloader['dataset'] = dataset
        dataloader = task_processor.build_dataloader(test_dataloader)

    # load the model of the backend
    model = task_processor.build_backend_model(args.model)
    destroy_model = model.destroy
    is_device_cpu = (args.device == 'cpu')

    runner = task_processor.build_test_runner(
        model,
        work_dir,
        log_file=args.log2file,
        show=args.show,
        show_dir=args.show_dir,
        wait_time=args.wait_time,
        interval=args.interval,
        dataloader=dataloader)

    if args.speed_test:
        with_sync = not is_device_cpu

        with TimeCounter.activate(
                warmup=args.warmup,
                log_interval=args.log_interval,
                with_sync=with_sync,
                file=args.log2file,
                batch_size=test_dataloader['batch_size']):
            runner.test()
    else:
        runner.test()
    # only effective when the backend requires explicit clean-up (e.g. Ascend)
    destroy_model()


if __name__ == '__main__':
    main()
    # def custom_filter(data_list):
    #     """Some filter operations cannot be combined with the previous operations, we must do it in the end.
    #        Whether to add this step depend on the final result.
    #     """
    #     for i in data_list.copy():
    #         if "python3.7" in i[0] and "mmengine" not in i[0]:
    #             data_list.remove(i)
    #     for n,i in enumerate(data_list):
    #         if "site-packages" in i[0]:
    #             data_list[n][0] = i[0].split("site-packages/")[-1]
    #     print("custom_filter success!")
    #     return data_list

    # # create a Trace object, telling it what to ignore
    # funcname = "main()"
    # ignoremods=["tqdm","matplotlib","PIL","tkinter"]
    # ignoredirs=["/home/sanbu/anaconda3/envs/mmlab2/lib/python3.7/site-packages/PIL",
    #             "/home/sanbu/anaconda3/envs/mmlab2/lib/python3.7/site-packages/matplotlib"]
    # filtermods=["_bootstrap","array_function","matplotlib","PIL"]
    # renamemods=["mmdetection3","envs"]

    # # build mytrace instance
    # my_trace = MyTrace(ignoremods,ignoredirs,filtermods,renamemods,filename = "trace_result.pkl",
    #                     funcname = funcname, if_annotation=True)
    # my_trace.run()
