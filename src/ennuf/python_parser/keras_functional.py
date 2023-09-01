#  (C) Crown Copyright, Met Office, 2023.
import ast
from _ast import Assign, Constant, Call, UnaryOp, Name
from typing import Any, Tuple

from ennuf.ml_model.layers.dense import Dense
from ennuf.ml_model.model import Model
from ennuf.utils.logger import LOGGER


def _get_line_numbers_string(node):
    return f'line {node.lineno}' if node.lineno == node.end_lineno else f'lines {node.lineno} to {node.end_lineno}'


class KerasFunctionalModelAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.stats = {"import": [], "from": []}
        self.variable_last_assignments = {}
        """A dictionary storing the last node a given name was assigned to"""
        self.model = Model()

    def visit_Assign(self, node: Assign) -> Any:
        """
        Called whenever an assignment statement is visited, ie something like somevariable = somevalue.
        This is the most important kind of statement for ML models created with the keras functional API,
        since layers are generally defined with statements like somevariable = somelayertype(args)(someothervariable).
        """
        target_id = self._get_target_id(node)
        value = node.value
        self.variable_last_assignments[target_id] = value
        if type(value) is Constant:
            # do stuff for if the target is being assigned to a Constant
            value: Constant
            const_value = value.value
            print(f'    Assignment of {const_value} to variable with name {target_id}')
            self.generic_visit(node)
            return
        if type(value) is Call:
            value: Call
            print(f'Assignment statement to target "{target_id}" whose value is a call statement discovered.')
            call, func_name = self._get_called_function(value)
            self._handle_assignment_to_call(node, call, func_name)

        self.generic_visit(node)

    def _get_called_function(self, node: Call) -> Tuple[Call, str]:
        """Attempts to get the Call instance and """
        # Now we attempt to get the name of the called function
        try:
            # First, we try statements of the form "somevariable = somefunction(args, kwargs)"
            call = node
            func_name: str = call.func.id
        except AttributeError:
            # Next, we try statements of the form "somevariable = ModuleClassOrObject.somefunction(args, kwargs)"
            try:
                call = node.func
                func_name: str = call.func.attr
            except AttributeError:
                # If we get something else, we should flag here that we couldn't understand it
                lines = _get_line_numbers_string(node)
                raise NotImplementedError(f'Unable to parse assignment to unknown call type at {lines}')
        return call, func_name

    def _handle_assignment_to_call(self, node: Assign, call: Call, func_name: str):
        match func_name:
            case 'Input':
                args = node.value.args
                kwargs = node.value.keywords

                print('This is a call to Input with arguments: ')
                for keyword in kwargs:
                    print(f'    keyword "{keyword.arg}" with value "{keyword.value.value}"')
            case 'Concatenate':
                args = call.args
                kwargs = call.keywords
                layer_input_list = node.value.args
                inputs_elts = layer_input_list[0].elts
                print('This is a call to Concatenate with arguments: ')
                for arg in args:
                    if type(arg) == UnaryOp:
                        # assume this is a subtraction
                        print(f'    arg with value "-{arg.operand.value}"')
                    else:
                        print(f'    arg with value "{arg.value}"')

                for keyword in kwargs:
                    print(f'    keyword "{keyword.arg}" with value "{keyword.value.value}"')
                print('The Concatenate layer is concatenating the following named layers:')
                for elt in inputs_elts:
                    layer_id = elt.id
                    print(f'    {layer_id}')
            case 'Dense':
                args = call.args
                kwargs = call.keywords
                layer_input_list = node.value.args
                input_layer = layer_input_list[0]
                print('This is a call to Dense with arguments: ')
                for arg in args:
                    if type(arg) == Name:
                        print(f'    arg with value {arg.id}')
                    elif type(arg) == UnaryOp:
                        # assume this is a subtraction
                        print(f'    arg with value "-{arg.operand.value}"')
                    else:
                        print(f'    arg with value "{arg.value}"')

                for keyword in kwargs:
                    val = keyword.value
                    if type(val) == Name:
                        val = val.id
                        actual_val = self.variable_last_assignments[val]
                        print(f'    keyword "{keyword.arg}" has actual value {actual_val}')
                    else:
                        val = val.value
                    print(f'    keyword "{keyword.arg}" with value "{val}"')
                print('The Dense layer is applied to this layer:')
                layer_id = input_layer.id
                print(f'    {layer_id}')
                self.model.layers.add(Dense())


    def _get_target_id(self, node: Assign):
        if len(node.targets) > 1:
            raise NotImplementedError(
                'have not implemented how to deal with statements where multiple variables are'
                ' assigned at once.'
            )
        target = node.targets[0]
        if type(target) is not Name:
            raise NotImplementedError('have not implemented how to deal with assigning a value to a tuple')
        return target.id
