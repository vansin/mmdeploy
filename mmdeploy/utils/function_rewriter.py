import logging

from mmcv.utils import Registry

from .register_utils import eval_with_import


# builder of register
def build_caller(func_name, backend, cfg, registry, **kwargs):
    # func_caller = registry.get(func_name + '@' + backend)
    func_caller = registry.module_dict[func_name + '@' + backend]
    assert func_caller is not None, '{} with {} not exist.'.format(
        func_name, backend)
    return func_caller(cfg, **kwargs)


# create register
FUNCTION_REWRITERS = Registry('func_rewriters', build_func=build_caller)


# caller wrapper
class FuncCaller(object):
    func_name = None
    backend = None
    func = None

    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        [setattr(self, k, v) for k, v in kwargs.items()]

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


# caller decorator
def register_rewriter(func_name, backend='default', **kwargs):

    def wrap(func):
        func_args = dict(func_name=func_name, backend=backend, func=func)
        func_args.update(kwargs)
        func_caller = type(func_name + '@' + backend, (FuncCaller, ),
                           func_args)
        return FUNCTION_REWRITERS.register_module()(func_caller)

    return wrap


FUNCTION_REWRITERS.register_rewriter = register_rewriter


def apply_rewriter(regist_func):

    def wrapper(*args, **kwargs):
        return regist_func(*args, **kwargs)

    return wrapper


class RewriterHook(object):

    def __init__(self, regist_name, cfg, **kwargs):
        func_name, backend = regist_name.split('@')
        self.func_name = func_name
        self.backend = backend
        self.regist_func = FUNCTION_REWRITERS.build(
            func_name, backend=self.backend, cfg=cfg, **kwargs)
        try:
            self.origin_func = eval_with_import(self.func_name)
        except Exception:
            self.origin_func = None
            logging.warning(
                'Can not found {}, function rewrite will not be applied'.
                format(self.func_name))

    def _set_func(self, rewrite_func):
        if self.origin_func is not None:
            # import necessary module
            split_path = self.func_name.split('.')
            for i in range(len(split_path), 0, -1):
                try:
                    exec('import {}'.format('.'.join(split_path[:i])))
                    break
                except Exception:
                    continue
            # assign function
            exec('{} = rewrite_func'.format(self.func_name))

    def __enter__(self):
        self._set_func(apply_rewriter(self.regist_func))

    def __exit__(self, type, val, tb):
        self._set_func(self.origin_func)


class RewriterContext(object):

    def __init__(self, cfg, backend='default', **kwargs):
        self.cfg = cfg
        func_backend_dict = {}
        for regist_name in FUNCTION_REWRITERS.module_dict:
            regist_func, regist_backend = regist_name.split('@')
            # only build `backend` or `default`
            if regist_backend not in [backend, 'default']:
                continue
            if regist_func not in func_backend_dict or func_backend_dict[
                    regist_func] == 'default':
                func_backend_dict[regist_func] = regist_backend

        self.hooks = [
            RewriterHook(k + '@' + v, cfg, **kwargs)
            for k, v in func_backend_dict.items()
        ]

    def __enter__(self):
        for hook in self.hooks:
            hook.__enter__()
        return self

    def __exit__(self, type, val, tb):
        for hook in self.hooks:
            hook.__exit__(type, val, tb)
