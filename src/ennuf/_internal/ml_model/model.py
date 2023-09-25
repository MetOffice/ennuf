#  (C) Crown Copyright, Met Office, 2023.
from pathlib import Path
from typing import List, Set, Iterable

import numpy as np

from ennuf._internal.config import CONFIG
from ennuf._internal.fortran import copy_neural_net_mod
from ennuf._internal.ml_model.base_layer import BaseLayer
from ennuf._internal.ml_model.layers import input_layer


class Model:
    layers: List[BaseLayer] = []
    """
    The model's layers. Note this is a list rather than a set, so that when displayed to a user the layers
    can appear easily in the same order they specified them; but this is *not* guarunteed to be the ordering of
    the layers of a sequential model.
    """

    @property
    def layer_dict(self):
        """A dict mapping the model's layers' names to the layer objects themselves"""
        return {layer.name: layer for layer in self.layers}

    def __init__(self, id_: str, long_name: str, output_names: List[str], dtype=np.float32, formatter=None):
        """

        Parameters
        ----------
        id_
         a valid fortran identifier, e.g. "ennuf_bcf"
        long_name
         a more descriptive name, e.g. "Bulk Cloud Fraction"
        output_names
         List of names of the network outputs, e.g. ["temperature_profile", "humidity_profile", "pressure_profile"]
        dtype
        formatter
         The fortran formatter for the model to use. Defaults to CONFIG.default_formatter() if None is provided.
        """
        self.id_ = id_
        self.long_name = long_name
        self.dtype = dtype
        self.output_names = output_names
        self._module_name = f'{self.id_}_mod'
        self.formatter = formatter if formatter is not None else CONFIG.default_formatter

    def __str__(self):
        description = f'An ML model with dtype {self.dtype} the following layers:\n'
        for layer in self.layers:
            description += str(layer) + ';\n'
        return description

    @property
    def inputs(self) -> Iterable[input_layer.InputLayer]:
        return filter(lambda layer: isinstance(layer, input_layer.InputLayer), self.layers)

    @property
    def outputs(self) -> Iterable[BaseLayer]:
        return filter(lambda layer: layer.name in self.output_names, self.layers)

    @property
    def layer_types_used(self) -> Set:
        return set(type(layer) for layer in self.layers)

    def to_fortran(self) -> str:
        return f'{self._fortran_file_head()}' \
               f'{self._fortran_module_head()}' \
               f'{self._fortran_subroutine()}' \
               f'{self._fortran_module_tail()}'

    def create_fortran_module(self, file: Path | str, overwrite: bool = False, include_neural_net_mod=True) -> None:
        mode = 'w' if overwrite else 'x'
        if not isinstance(file, Path):
            file = Path(file)
        with open(file, mode) as module_file:
            module_file.write(self.to_fortran())
        if include_neural_net_mod:
            dest_dir = file.parent
            copy_neural_net_mod(dest_dir)

    def _fortran_file_head(self) -> str:
        """Returns text that goes at the top of the fortran file"""
        required_header = self.formatter.required_file_header()
        header_comment = self.formatter.format_line(f'! Easy Neural Networks in the Um in Fortran: {self.long_name}')
        return f'{required_header}' \
               f'\n' \
               f'{header_comment}'

    def _fortran_module_head(self) -> str:
        """Returns text that begins the fortran module this_model_mod"""
        module_stmt = self.formatter.format_line(f'MODULE {self._module_name}')
        implicit_stmt = self.formatter.format_line('IMPLICIT NONE')
        required_imports = self.formatter.required_module_imports()
        required_declarations = self.formatter.required_module_declarations(self._module_name)
        contains_stmt = self.formatter.format_line('CONTAINS')
        return f'{module_stmt}' \
               f'{required_imports}' \
               f'{implicit_stmt}' \
               f'{required_declarations}' \
               f'{contains_stmt}'

    def _fortran_subroutine(self) -> str:
        """Returns text that begins the fortran subroutine this_model"""
        subroutine_name = self.id_
        # build the SUBROUTINE statement:
        arg_list = []
        for input_layer_ in self.inputs:
            arg_list.append(input_layer_.name)
        for output_name in self.output_names:
            # since output names have y_ prepended elsewhere to distinguish them from weights.
            arg_list.append(f'y_{output_name}')
        arg_str = ''
        for arg in arg_list:
            if arg_str:
                arg_str = f'{arg_str}, {arg}'
            else:
                arg_str = f'{arg}'
        subroutine_stmt = self.formatter.format_line(f'SUBROUTINE {subroutine_name}({arg_str})')
        # build the imports of neural network stuff:
        layer_types_to_import = ''
        for layer_type in self.layer_types_used:
            if layer_type.fortran_id():
                if layer_types_to_import:
                    layer_types_to_import = f'{layer_types_to_import} ,{layer_type.fortran_id()}'
                else:
                    layer_types_to_import = layer_type.fortran_id()
        import_stmt = self.formatter.format_line(f'USE neural_net_mod, ONLY: {layer_types_to_import}\n')
        required_imports_stmt = self.formatter.required_subroutine_imports()
        implicit_stmt = self.formatter.format_line('IMPLICIT NONE\n')
        required_decls_stmt = self.formatter.required_subroutine_declarations(subroutine_name)
        layer_typedecl_stmts = ''
        layer_init_stmts = ''
        for layer in self.layers:
            # TODO: make this depend on model dtype field
            layer_typedecls = layer.get_fortran_type_declaration(self.formatter.default_dtype)
            layer_typedecl_stmts = f'{layer_typedecl_stmts}{layer_typedecls}\n'
            layer_inits = layer.get_fortran_data_initialisation()
            layer_init_stmts = f'{layer_init_stmts}{layer_inits}\n'
        required_opening_stmts = self.formatter.required_subroutine_opening_actions()
        main_body = ''
        for layer in self.layers:
            if layer.input_name is not None:
                main_body = f'{main_body}{layer.get_fortran_layer_subroutine_call_stmt()}\n'
        required_closing_stmts = self.formatter.required_subroutine_closing_actions()
        return_stmt = 'RETURN'
        end_subroutine_stmt = f'END SUBROUTINE {subroutine_name}'
        return f'{subroutine_stmt}' \
               f'{required_imports_stmt}' \
               f'{import_stmt}' \
               f'{implicit_stmt}' \
               f'{required_decls_stmt}' \
               f'{layer_typedecl_stmts}' \
               f'{layer_init_stmts}\n' \
               f'{required_opening_stmts}\n' \
               f'{main_body}\n' \
               f'{required_closing_stmts}\n' \
               f'{return_stmt}\n' \
               f'{end_subroutine_stmt}\n'

    def _fortran_module_tail(self) -> str:
        """Returns text that ends the fortran module this_model_mod."""
        return f'END MODULE {self._module_name}\n'
