# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import importlib
import logging
import os
import os.path as osp
import pickle
import trace
from functools import partial
from typing import List

import mmengine
import pandas as pd
import torch.multiprocessing as mp
from torch.multiprocessing import Process, set_start_method

from mmdeploy.apis import (create_calib_input_data, extract_model,
                           get_predefined_partition_cfg, torch2onnx,
                           torch2torchscript, visualize_model)
from mmdeploy.apis.core import PIPELINE_MANAGER
from mmdeploy.backend.sdk.export_info import export2SDK
from mmdeploy.utils import (IR, Backend, get_backend, get_calib_filename,
                            get_ir_config, get_model_inputs,
                            get_partition_config, get_root_logger, load_config,
                            target_wrapper)


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
    parser = argparse.ArgumentParser(description='Export model to backends.')
    parser.add_argument('deploy_cfg', help='deploy config path')
    parser.add_argument('model_cfg', help='model config path')
    parser.add_argument('checkpoint', help='model checkpoint path')
    parser.add_argument('img', help='image used to convert model model')
    parser.add_argument(
        '--test-img', default=None, help='image used to test model')
    parser.add_argument(
        '--work-dir',
        default=os.getcwd(),
        help='the dir to save logs and models')
    parser.add_argument(
        '--calib-dataset-cfg',
        help='dataset config path used to calibrate in int8 mode. If not \
            specified, it will use "val" dataset in model config instead.',
        default=None)
    parser.add_argument(
        '--device', help='device used for conversion', default='cpu')
    parser.add_argument(
        '--log-level',
        help='set log level',
        default='INFO',
        choices=list(logging._nameToLevel.keys()))
    parser.add_argument(
        '--show', action='store_true', help='Show detection outputs')
    parser.add_argument(
        '--dump-info', action='store_true', help='Output information for SDK')
    parser.add_argument(
        '--quant-image-dir',
        default=None,
        help='Image directory for quantize model.')
    parser.add_argument(
        '--quant', action='store_true', help='Quantize model to low bit.')
    parser.add_argument(
        '--uri',
        default='192.168.1.1:60000',
        help='Remote ipv4:port or ipv6:port for inference on edge device.')
    args = parser.parse_args()
    return args


def create_process(name, target, args, kwargs, ret_value=None):
    logger = get_root_logger()
    logger.info(f'{name} start.')
    log_level = logger.level

    wrap_func = partial(target_wrapper, target, log_level, ret_value)

    process = Process(target=wrap_func, args=args, kwargs=kwargs)
    process.start()
    process.join()

    if ret_value is not None:
        if ret_value.value != 0:
            logger.error(f'{name} failed.')
            exit(1)
        else:
            logger.info(f'{name} success.')


def torch2ir(ir_type: IR):
    """Return the conversion function from torch to the intermediate
    representation.

    Args:
        ir_type (IR): The type of the intermediate representation.
    """
    if ir_type == IR.ONNX:
        return torch2onnx
    elif ir_type == IR.TORCHSCRIPT:
        return torch2torchscript
    else:
        raise KeyError(f'Unexpected IR type {ir_type}')


def main():
    args = parse_args()
    set_start_method('spawn', force=True)
    logger = get_root_logger()
    log_level = logging.getLevelName(args.log_level)
    logger.setLevel(log_level)

    pipeline_funcs = [
        torch2onnx, torch2torchscript, extract_model, create_calib_input_data
    ]
    PIPELINE_MANAGER.enable_multiprocess(True, pipeline_funcs)
    PIPELINE_MANAGER.set_log_level(log_level, pipeline_funcs)

    deploy_cfg_path = args.deploy_cfg
    model_cfg_path = args.model_cfg
    checkpoint_path = args.checkpoint
    quant = args.quant
    quant_image_dir = args.quant_image_dir

    # load deploy_cfg
    deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)

    # create work_dir if not
    mmengine.mkdir_or_exist(osp.abspath(args.work_dir))

    if args.dump_info:
        export2SDK(
            deploy_cfg,
            model_cfg,
            args.work_dir,
            pth=checkpoint_path,
            device=args.device)

    ret_value = mp.Value('d', 0, lock=False)

    # convert to IR
    ir_config = get_ir_config(deploy_cfg)
    ir_save_file = ir_config['save_file']
    ir_type = IR.get(ir_config['type'])
    torch2ir(ir_type)(
        args.img,
        args.work_dir,
        ir_save_file,
        deploy_cfg_path,
        model_cfg_path,
        checkpoint_path,
        device=args.device)

    # convert backend
    ir_files = [osp.join(args.work_dir, ir_save_file)]

    # partition model
    partition_cfgs = get_partition_config(deploy_cfg)

    if partition_cfgs is not None:

        if 'partition_cfg' in partition_cfgs:
            partition_cfgs = partition_cfgs.get('partition_cfg', None)
        else:
            assert 'type' in partition_cfgs
            partition_cfgs = get_predefined_partition_cfg(
                deploy_cfg, partition_cfgs['type'])

        origin_ir_file = ir_files[0]
        ir_files = []
        for partition_cfg in partition_cfgs:
            save_file = partition_cfg['save_file']
            save_path = osp.join(args.work_dir, save_file)
            start = partition_cfg['start']
            end = partition_cfg['end']
            dynamic_axes = partition_cfg.get('dynamic_axes', None)

            extract_model(
                origin_ir_file,
                start,
                end,
                dynamic_axes=dynamic_axes,
                save_file=save_path)

            ir_files.append(save_path)

    # calib data
    calib_filename = get_calib_filename(deploy_cfg)
    if calib_filename is not None:
        calib_path = osp.join(args.work_dir, calib_filename)
        create_calib_input_data(
            calib_path,
            deploy_cfg_path,
            model_cfg_path,
            checkpoint_path,
            dataset_cfg=args.calib_dataset_cfg,
            dataset_type='val',
            device=args.device)

    backend_files = ir_files
    # convert backend
    backend = get_backend(deploy_cfg)
    if backend == Backend.TENSORRT:
        model_params = get_model_inputs(deploy_cfg)
        assert len(model_params) == len(ir_files)

        from mmdeploy.apis.tensorrt import is_available as trt_is_available
        assert trt_is_available(
        ), 'TensorRT is not available,' \
            + ' please install TensorRT and build TensorRT custom ops first.'

        from mmdeploy.apis.tensorrt import onnx2tensorrt
        PIPELINE_MANAGER.enable_multiprocess(True, [onnx2tensorrt])
        PIPELINE_MANAGER.set_log_level(log_level, [onnx2tensorrt])

        backend_files = []
        for model_id, model_param, onnx_path in zip(
                range(len(ir_files)), model_params, ir_files):
            onnx_name = osp.splitext(osp.split(onnx_path)[1])[0]
            save_file = model_param.get('save_file', onnx_name + '.engine')

            partition_type = 'end2end' if partition_cfgs is None \
                else onnx_name
            onnx2tensorrt(
                args.work_dir,
                save_file,
                model_id,
                deploy_cfg_path,
                onnx_path,
                device=args.device,
                partition_type=partition_type)

            backend_files.append(osp.join(args.work_dir, save_file))

    elif backend == Backend.NCNN:
        from mmdeploy.apis.ncnn import is_available as is_available_ncnn

        if not is_available_ncnn():
            logger.error('ncnn support is not available, please make sure \
                1) `mmdeploy_onnx2ncnn` existed in `PATH` \
                2) python import ncnn success')
            exit(1)

        import mmdeploy.apis.ncnn as ncnn_api
        from mmdeploy.apis.ncnn import get_output_model_file

        PIPELINE_MANAGER.set_log_level(log_level, [ncnn_api.from_onnx])

        backend_files = []
        for onnx_path in ir_files:
            model_param_path, model_bin_path = get_output_model_file(
                onnx_path, args.work_dir)
            onnx_name = osp.splitext(osp.split(onnx_path)[1])[0]
            ncnn_api.from_onnx(onnx_path, osp.join(args.work_dir, onnx_name))

            if quant:
                from onnx2ncnn_quant_table import get_table

                from mmdeploy.apis.ncnn import get_quant_model_file, ncnn2int8

                deploy_cfg, model_cfg = load_config(deploy_cfg_path,
                                                    model_cfg_path)
                quant_onnx, quant_table, quant_param, quant_bin = get_quant_model_file(  # noqa: E501
                    onnx_path, args.work_dir)

                create_process(
                    'ncnn quant table',
                    target=get_table,
                    args=(onnx_path, deploy_cfg, model_cfg, quant_onnx,
                          quant_table, quant_image_dir, args.device),
                    kwargs=dict(),
                    ret_value=ret_value)

                create_process(
                    'ncnn_int8',
                    target=ncnn2int8,
                    args=(model_param_path, model_bin_path, quant_table,
                          quant_param, quant_bin),
                    kwargs=dict(),
                    ret_value=ret_value)
                backend_files += [quant_param, quant_bin]
            else:
                backend_files += [model_param_path, model_bin_path]

    elif backend == Backend.SNPE:
        from mmdeploy.apis.snpe import is_available as is_available

        if not is_available():
            logger.error('snpe support is not available, please check \
                1) `snpe-onnx-to-dlc` existed in `PATH` 2) snpe only support \
                    ubuntu18.04')
            exit(1)

        import mmdeploy.apis.snpe as snpe_api
        from mmdeploy.apis.snpe import get_env_key, get_output_model_file

        if get_env_key() not in os.environ:
            os.environ[get_env_key()] = args.uri

        PIPELINE_MANAGER.set_log_level(log_level, [snpe_api.from_onnx])

        backend_files = []
        for onnx_path in ir_files:
            dlc_path = get_output_model_file(onnx_path, args.work_dir)
            onnx_name = osp.splitext(osp.split(onnx_path)[1])[0]
            snpe_api.from_onnx(onnx_path, osp.join(args.work_dir, onnx_name))
            backend_files = [dlc_path]

    elif backend == Backend.OPENVINO:
        from mmdeploy.apis.openvino import \
            is_available as is_available_openvino
        assert is_available_openvino(), \
            'OpenVINO is not available, please install OpenVINO first.'

        import mmdeploy.apis.openvino as openvino_api
        from mmdeploy.apis.openvino import (get_input_info_from_cfg,
                                            get_mo_options_from_cfg,
                                            get_output_model_file)

        PIPELINE_MANAGER.set_log_level(log_level, [openvino_api.from_onnx])

        openvino_files = []
        for onnx_path in ir_files:
            model_xml_path = get_output_model_file(onnx_path, args.work_dir)
            input_info = get_input_info_from_cfg(deploy_cfg)
            output_names = get_ir_config(deploy_cfg).output_names
            mo_options = get_mo_options_from_cfg(deploy_cfg)
            openvino_api.from_onnx(onnx_path, args.work_dir, input_info,
                                   output_names, mo_options)
            openvino_files.append(model_xml_path)
        backend_files = openvino_files

    elif backend == Backend.PPLNN:
        from mmdeploy.apis.pplnn import is_available as is_available_pplnn
        assert is_available_pplnn(), \
            'PPLNN is not available, please install PPLNN first.'

        from mmdeploy.apis.pplnn import from_onnx

        pplnn_pipeline_funcs = [from_onnx]
        PIPELINE_MANAGER.set_log_level(log_level, pplnn_pipeline_funcs)

        pplnn_files = []
        for onnx_path in ir_files:
            algo_file = onnx_path.replace('.onnx', '.json')
            model_inputs = get_model_inputs(deploy_cfg)
            assert 'opt_shape' in model_inputs, 'Expect opt_shape ' \
                'in deploy config for PPLNN'
            # PPLNN accepts only 1 input shape for optimization,
            # may get changed in the future
            input_shapes = [model_inputs.opt_shape]
            algo_prefix = osp.splitext(algo_file)[0]
            from_onnx(
                onnx_path,
                algo_prefix,
                device=args.device,
                input_shapes=input_shapes)
            pplnn_files += [onnx_path, algo_file]
        backend_files = pplnn_files

    elif backend == Backend.RKNN:
        from mmdeploy.apis.rknn import is_available as rknn_is_available
        assert rknn_is_available(
        ), 'RKNN is not available, please install RKNN first.'

        from mmdeploy.apis.rknn import onnx2rknn
        PIPELINE_MANAGER.enable_multiprocess(True, [onnx2rknn])
        PIPELINE_MANAGER.set_log_level(logging.INFO, [onnx2rknn])

        backend_files = []
        for model_id, onnx_path in zip(range(len(ir_files)), ir_files):
            pre_fix_name = osp.splitext(osp.split(onnx_path)[1])[0]
            output_path = osp.join(args.work_dir, pre_fix_name + '.rknn')
            import tempfile
            dataset_file = tempfile.NamedTemporaryFile(suffix='.txt').name
            with open(dataset_file, 'w') as f:
                f.writelines([osp.abspath(args.img)])
            onnx2rknn(
                onnx_path,
                output_path,
                deploy_cfg_path,
                dataset_file=dataset_file)

            backend_files.append(output_path)
    elif backend == Backend.ASCEND:
        from mmdeploy.apis.ascend import from_onnx

        ascend_pipeline_funcs = [from_onnx]
        PIPELINE_MANAGER.set_log_level(log_level, ascend_pipeline_funcs)

        model_inputs = get_model_inputs(deploy_cfg)

        om_files = []
        for model_id, onnx_path in enumerate(ir_files):
            om_path = osp.splitext(onnx_path)[0] + '.om'
            from_onnx(onnx_path, args.work_dir, model_inputs[model_id])
            om_files.append(om_path)
        backend_files = om_files

        if args.dump_info:
            from mmdeploy.backend.ascend import update_sdk_pipeline
            update_sdk_pipeline(args.work_dir)

    elif backend == Backend.COREML:
        from mmdeploy.apis.coreml import from_torchscript, get_model_suffix
        coreml_pipeline_funcs = [from_torchscript]
        PIPELINE_MANAGER.set_log_level(log_level, coreml_pipeline_funcs)
        model_inputs = get_model_inputs(deploy_cfg)
        coreml_files = []
        for model_id, torchscript_path in enumerate(ir_files):
            torchscript_name = osp.splitext(osp.split(torchscript_path)[1])[0]
            output_file_prefix = osp.join(args.work_dir, torchscript_name)
            convert_to = deploy_cfg.backend_config.convert_to
            from_torchscript(torchscript_path, output_file_prefix,
                             ir_config.input_names, ir_config.output_names,
                             model_inputs[model_id].input_shapes, convert_to)
            suffix = get_model_suffix(convert_to)
            coreml_files.append(output_file_prefix + suffix)
        backend_files = coreml_files

    if args.test_img is None:
        args.test_img = args.img

    extra = dict(
        backend=backend,
        output_file=osp.join(args.work_dir, f'output_{backend.value}.jpg'),
        show_result=args.show)
    if backend == Backend.SNPE:
        extra['uri'] = args.uri

    # get backend inference result, try render
    create_process(
        f'visualize {backend.value} model',
        target=visualize_model,
        args=(model_cfg_path, deploy_cfg_path, backend_files, args.test_img,
              args.device),
        kwargs=extra,
        ret_value=ret_value)

    # get pytorch model inference result, try visualize if possible
    create_process(
        'visualize pytorch model',
        target=visualize_model,
        args=(model_cfg_path, deploy_cfg_path, [checkpoint_path],
              args.test_img, args.device),
        kwargs=dict(
            backend=Backend.PYTORCH,
            output_file=osp.join(args.work_dir, 'output_pytorch.jpg'),
            show_result=args.show),
        ret_value=ret_value)
    logger.info('All process success.')


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
