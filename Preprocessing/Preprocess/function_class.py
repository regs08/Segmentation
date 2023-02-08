"""
base class for our preprocess, takes in a list of functions and kwargs will check to make sure all the functions have
fulfilled args from the kwargs

"""
import inspect


class FunctionClass:
    def __init__(self,
                 functions,
                 **kwargs):

        self.__dict__.update(kwargs)
        self.functions = functions

    def prepare(self):
        """
        updates the dict adds in any other args made e.g makling a save dir as in the preprocess sub class

        :return:
        """
        self.args = self.get_args_from_self()
        self.function_args = self.get_args_from_all_functions()
        self.__dict__.update()


    def get_args_from_all_functions(self):
        """
        wrapper for iterating through the list of functions

        :return: {func_name_1: {arg_1: val_1, arg_2: val_2} ...
        """
        all_func_args = dict()
        for f in self.functions:
            all_func_args[f] = self.get_args_from_function(self.functions[f])
        return all_func_args

    def get_args_from_function(self, function):
        """
        takes in a function returns a dict where the keys are the arg name and the value is the default value, None if
        there is no default value
        :param function:
        :return: function params: {'arg_name' : arg_value; default=None}
        """
        function_args = dict()
        default_args = inspect_and_get_defaults_of_function(function)

        #overriding default values set
        for p in default_args:
            if default_args[p] and p not in self.args.keys():
                function_args[p] = default_args[p]
                continue
            assert p in self.args.keys(), f'Var name {p} not found in class args: {list(self.args.keys())}'
            function_args[p] = self.args[p]
        return function_args

    def get_args_from_self(self):
        """
        gets args from self to compare and check with the func args
        :return:
        """
        return {key: value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}

    def run_function(self, f):
        self.functions[f](**self.function_args[f])

    def run_all_functions(self):
        for f in self.functions:
            self.functions[f](**self.function_args[f])

"""
Utils
"""


def inspect_and_get_defaults_of_function(function):
    """
    takes in a function gets its args and returns the default values  of args returns None if no default is set
    :param function:
    :return: args: dict containing arg values, None if there is no default
    """
    function_argspec = inspect.getfullargspec(function)
    function_args = function_argspec.args
    function_defaults = function_argspec.defaults
    args = dict(zip(function_args, [None] * len(function_args)))

    if function_defaults:
        """
        inspecting our functions params. 
        """
        sig = inspect.signature(function)
        for p in sig.parameters:
            default_value_of_arg = sig.parameters[p].default
            if default_value_of_arg != inspect._empty:
                args[p] = default_value_of_arg
    return args