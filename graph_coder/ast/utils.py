import ast
import typing


empty_token = "[PAD]"


_const_types = {
    ast.Num: (int, float, complex),
    ast.Str: (str,),
    ast.Bytes: (bytes,),
    ast.NameConstant: (type(None), bool),
    ast.Ellipsis: (type(...),),
}


def wrap(node: typing.Union[ast.AST, str, int, float, bool, bytes]) -> ast.AST:
    if isinstance(node, ast.AST):
        return node

    for t, v in _const_types.items():
        if isinstance(node, v):
            if isinstance(node, (str, bytes)):
                return t(s=node)
            if isinstance(node, (int, float, complex)):
                return t(n=node)

            return t(value=node)

    return node
