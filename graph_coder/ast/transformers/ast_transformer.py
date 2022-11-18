import ast
import typing

from graph_coder.ast.context import Context
from graph_coder.ast.edge import Edge
from graph_coder.ast.edge_attributes import EdgeAttributes
from graph_coder.ast.node import Node
from graph_coder.ast.node_attributes import NodeAttributes
from graph_coder.ast.nodes.class_def import ClassDef
from graph_coder.ast.nodes.key_value import KeyValue
from graph_coder.ast.transformers.class_name_transformer import __class_name__
from graph_coder.ast.transformers.function_name_transformer import __fn_name__
from graph_coder.ast.utils import empty_token, wrap


def parse_type(body: ast.AST, slice: typing.Iterable[ast.AST] = None) -> str:
    res = ""
    if isinstance(body, ast.Attribute):
        res += f"{body.value.id}.{body.attr}"
    if isinstance(body, ast.Name):
        res += body.id
    if isinstance(body, ast.Subscript):
        if hasattr(body.slice, "eilts"):
            res += parse_type(ast.Name(id="List"), body.slice.eilts)
        else:
            res += parse_type(body.slice)
    if isinstance(body, (ast.List, ast.Tuple)):
        res += parse_type(ast.Name(id="List"), body.elts)
    if isinstance(body, ast.Constant):
        if body.kind is not None:
            res += body.kind
        else:
            res += "const"
    if isinstance(body, KeyValue):
        res += str(body)

    if slice is not None:
        res += f"[{','.join([parse_type(s) for s in slice])}]"

    if res == "":
        raise NotImplementedError("Unknown type: " + body.__class__.__name__)

    return res


__simple_nodes__ = (
    ast.Constant,
    ast.Break,
    ast.Continue,
    ast.Or,
    ast.Pass,
    ast.Mod,
    ast.Sub,
    ast.Div,
    ast.FloorDiv,
    ast.Mult,
    ast.MatMult,
    ast.Add,
    ast.UAdd,
    ast.And,
    ast.Pow,
    ast.Eq,
    ast.NotEq,
    ast.In,
    ast.Not,
    ast.LShift,
    ast.LtE,
    ast.Lt,
    ast.Gt,
    ast.GtE,
    ast.Is,
    ast.BitAnd,
    ast.BitOr,
    ast.BitXor,
)
__skip_nodes__ = (ast.Load, ast.Store, ast.Del)
__edges__ = "__edges__"


class ASTransformer(ast.NodeTransformer):
    def __init__(
        self, ctx: Context, simple_nodes=__simple_nodes__, skip_nodes=__skip_nodes__
    ):
        self.ctx = ctx
        self.simple_nodes = simple_nodes
        self.skip_nodes = skip_nodes

    def visit(self, node: ast.AST) -> typing.Any:
        if isinstance(node, self.simple_nodes):
            return self.simple_visit(node)
        elif isinstance(node, self.skip_nodes):
            return self.generic_visit(node)
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, None)
        if visitor is None:
            raise NotImplementedError("Unknown node: " + ast.dump(node))
        return visitor(node)

    def visit_SetComp(self, node: ast.SetComp) -> typing.Any:
        edges = [
            Edge(source_right=gen, relation="generator") for gen in node.generators
        ]

        del node.generators

        return self.post_hook(
            node,
            node=Node.from_ast_body(self.ctx.id, node.elt, self.ctx.depth),
            edges=edges,
        )

    def visit_Starred(self, node: ast.Starred) -> typing.Any:
        return self.post_hook(
            node,
            node=Node.from_ast_body(self.ctx.id, node.value, self.ctx.depth),
        )

    def visit_Await(self, node: ast.Await) -> typing.Any:
        return self.post_hook(
            node,
            node=Node.from_ast_body(self.ctx.id, node.value, self.ctx.depth),
        )

    def visit_DictComp(self, node: ast.DictComp) -> typing.Any:
        edges = [
            Edge(source_right=node.key, relation="key"),
            Edge(source_right=node.value, relation="value"),
        ]

        for gen in node.generators:
            edges.append(Edge(source_right=gen.iter, relation="iterates"))
            edges.append(Edge(source_right=gen.target, relation="target"))
            edges.extend([Edge(source_right=b, relation="body") for b in gen.ifs])

        del node.generators

        return self.post_hook(
            node,
            node=Node.from_ast_body(self.ctx.id, node, self.ctx.depth),
            edges=edges,
        )

    def visit_Delete(self, node: ast.Delete) -> typing.Any:
        edges = [
            Edge(source_right=target, relation="delete") for target in node.targets
        ]

        return self.post_hook(
            node,
            node=Node.from_ast_body(self.ctx.id, node, self.ctx.depth),
            edges=edges,
        )

    def visit_ClassDef(self, body: ClassDef) -> typing.Any:
        edges = [
            Edge(source_right=body.slf, relation="self"),
            Edge(source_right=body.cls, relation="cls"),
        ]

        for prop in body.properties:
            edges.append(Edge(source_right=prop, relation="property"))

        for inner in body.body:
            if isinstance(inner, ast.FunctionDef):
                edges.append(Edge(source_right=inner, relation="method"))

        self.ctx[body.name] = Node.from_ast_body(
            self.ctx.id, body, self.ctx.depth, NodeAttributes(type_name=body.name)
        )

        return self.post_hook(
            body,
            self.ctx[body.name],
            edges,
        )

    def visit_Tuple(self, body: ast.Tuple) -> typing.Any:
        edges = [Edge(source_right=c, relation="tuple") for c in body.elts]

        return self.post_hook(
            body, Node.from_ast_body(self.ctx.id, body, self.ctx.depth), edges
        )

    def visit_If(self, body: ast.If) -> typing.Any:
        edges = [Edge(source_right=body.test, relation="test")]
        edges.extend([Edge(source_right=b, relation="body") for b in body.body])
        edges.extend([Edge(source_right=b, relation="else") for b in body.orelse])

        return self.post_hook(
            body, Node.from_ast_body(self.ctx.id, body, self.ctx.depth), edges
        )

    def visit_For(self, body: ast.For) -> typing.Any:
        edges = [
            Edge(source_right=body.iter, relation="iterates"),
            Edge(source_right=body.target, relation="target"),
        ]
        edges.extend([Edge(source_right=b, relation="body") for b in body.body])
        edges.extend([Edge(source_right=b, relation="else") for b in body.orelse])

        return self.post_hook(
            body, Node.from_ast_body(self.ctx.id, body, self.ctx.depth), edges
        )

    def visit_AnnAssign(self, body: ast.AnnAssign) -> typing.Any:
        edges = []

        if body.value is not None:
            edges.append(
                Edge(source_right=body.value, relation="assigns"),
            )

        if body.annotation is not None:
            annotation = parse_type(body.annotation)
        else:
            annotation = None

        del body.annotation

        if body.target.__str__() not in self.ctx:
            node = self.ctx[body.target.__str__()] = Node.from_ast_body(
                self.ctx.id,
                body.target,
                self.ctx.depth,
                NodeAttributes(
                    type_name=annotation if annotation is not None else empty_token
                ),
            )
        else:
            node = self.ctx[body.target.__str__()]

        return self.post_hook(
            body,
            node=node,
            edges=edges,
        )

    def visit_FormattedValue(self, body: ast.FormattedValue) -> typing.Any:
        return self.post_hook(
            body,
            node=Node.from_ast_body(self.ctx.id, body.value, self.ctx.depth),
            edges=[Edge(source_right=body.value, relation="formats")],
        )

    def visit_Lambda(self, body: ast.Lambda) -> typing.Any:
        edges = []
        if body.args.args is not None:
            for arg in body.args.args:
                edges.append(
                    Edge(source_right=arg, relation="named_arg", attr=EdgeAttributes())
                )
        if body.args.vararg is not None:
            edges.append(
                Edge(
                    source_right=body.args.vararg,
                    relation="vararg",
                    attr=EdgeAttributes(),
                )
            )
        if body.args.kwarg is not None:
            edges.append(
                Edge(
                    source_right=body.args.kwarg,
                    relation="kwarg",
                    attr=EdgeAttributes(),
                )
            )
        edges.append(
            Edge(source_right=body.body, relation="body", attr=EdgeAttributes())
        )

        return self.post_hook(
            body,
            node=Node.from_ast_body(self.ctx.id, body, self.ctx.depth),
            edges=edges,
        )

    def visit_Raise(self, body: ast.Raise) -> typing.Any:
        edges = []

        if body.exc is not None:
            edges.append(Edge(source_right=body.exc, relation="raises"))

        if body.cause is not None:
            edges.append(Edge(source_right=body.cause, relation="cause"))

        return self.post_hook(
            body,
            node=Node.from_ast_body(self.ctx.id, body, self.ctx.depth),
            edges=edges,
        )

    def visit_Set(self, node: ast.Set) -> typing.Any:
        edges = [Edge(source_right=c, relation="set") for c in node.elts]

        return self.post_hook(
            node, Node.from_ast_body(self.ctx.id, node, self.ctx.depth), edges
        )

    def visit_JoinedStr(self, body: ast.JoinedStr) -> typing.Any:
        edges = []

        for c in body.values:
            if isinstance(c, ast.FormattedValue):
                edges.append(Edge(source_right=c.value, relation="concat"))
                body.values[body.values.index(c)] = c.value
            else:
                edges.append(Edge(source_right=c, relation="concat"))

        return self.post_hook(
            body,
            node=Node.from_ast_body(self.ctx.id, body, self.ctx.depth),
            edges=edges,
        )

    def visit_Compare(self, node: ast.Compare) -> typing.Any:
        edges = []

        for i, op in enumerate(node.ops):
            edges.append(
                Edge(source_right=node.comparators[i], relation=op.__class__.__name__)
            )

        del node.ops
        del node.comparators

        return self.post_hook(
            node,
            node=Node.from_ast_body(self.ctx.id, node.left, self.ctx.depth),
            edges=edges,
        )

    def visit_NamedExpr(self, node: ast.NamedExpr) -> typing.Any:
        return self.post_hook(
            node,
            node=Node.from_ast_body(self.ctx.id, node.target, self.ctx.depth),
            edges=[Edge(source_right=node.value, relation="assigns")],
        )

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> typing.Any:
        return self.visit_FunctionDef(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> typing.Any:
        return self.visit_For(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> typing.Any:
        return self.visit_With(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> typing.Any:
        edges = []

        for i, elt in enumerate(node.generators):
            edges.append(Edge(source_right=elt.iter, relation="iterates"))
            edges.append(Edge(source_right=elt.target, relation="target"))
            edges.extend([Edge(source_right=b, relation="if") for b in elt.ifs])

        del node.generators

        return self.post_hook(
            node,
            node=Node.from_ast_body(self.ctx.id, node, self.ctx.depth),
            edges=edges,
        )

    def visit_While(self, node: ast.While) -> typing.Any:
        edges = [
            Edge(source_right=node.test, relation="test"),
        ]

        edges.extend([Edge(source_right=b, relation="body") for b in node.body])
        edges.extend([Edge(source_right=b, relation="else") for b in node.orelse])

        return self.post_hook(
            node,
            node=Node.from_ast_body(self.ctx.id, node, self.ctx.depth),
            edges=edges,
        )

    def visit_Slice(self, node: ast.Slice) -> typing.Any:
        edges = []

        if node.lower is not None:
            edges.append(Edge(source_right=node.lower, relation="lower"))

        if node.upper is not None:
            edges.append(Edge(source_right=node.upper, relation="upper"))

        if node.step is not None:
            edges.append(Edge(source_right=node.step, relation="step"))

        return self.post_hook(
            node,
            node=Node.from_ast_body(self.ctx.id, node, self.ctx.depth),
            edges=edges,
        )

    def visit_ListComp(self, node: ast.ListComp) -> typing.Any:
        edges = []

        for gen in node.generators:
            edges.append(Edge(source_right=gen.iter, relation="iterates"))
            edges.append(Edge(source_right=gen.target, relation="target"))
            edges.extend([Edge(source_right=ff, relation="if") for ff in gen.ifs])

        del node.generators

        edges.append(Edge(source_right=node.elt, relation="element"))

        return self.post_hook(
            node,
            node=Node.from_ast_body(self.ctx.id, node, self.ctx.depth),
            edges=edges,
        )

    def visit_Global(self, node: ast.Global) -> typing.Any:
        return self.post_hook(
            node,
            node=Node.from_ast_body(self.ctx.id, node, self.ctx.depth),
            edges=[
                Edge(
                    source_right=ast.Name(id="global." + n, ctx=ast.Load()),
                    relation="global",
                )
                for n in node.names
            ],
        )

    def visit_With(self, node: ast.With) -> typing.Any:
        edges = []

        for item in node.items:
            if item.optional_vars is not None:
                edges.append(Edge(source_right=item.optional_vars, relation="as"))

        del node.items

        edges.extend([Edge(source_right=b, relation="body") for b in node.body])

        return self.post_hook(
            node,
            node=Node.from_ast_body(self.ctx.id, node, self.ctx.depth),
            edges=edges,
        )

    def visit_IfExp(self, node: ast.IfExp) -> typing.Any:
        edges = [
            Edge(source_right=node.test, relation="test"),
            Edge(source_right=node.body, relation="body"),
            Edge(source_right=node.orelse, relation="else"),
        ]

        return self.post_hook(
            node,
            node=Node.from_ast_body(self.ctx.id, node, self.ctx.depth),
            edges=edges,
        )

    def visit_Assert(self, node: ast.Assert) -> typing.Any:
        edges = []

        if node.msg is not None:
            edges.append(Edge(source_right=node.msg, relation="msg"))

        return self.post_hook(
            node,
            node=Node.from_ast_body(self.ctx.id, node, self.ctx.depth),
            edges=edges,
        )

    def visit_Call(self, body: ast.Call) -> typing.Any:
        edges = []
        if len(body.keywords) != 0:
            edges.extend(
                [Edge(source_right=k, relation="keyword") for k in body.keywords]
            )

        for i, target in enumerate(body.args):
            if isinstance(target, ast.Name):
                rel = Edge(
                    source_right=target,
                    relation="calls",
                    attr=EdgeAttributes(),
                )
                edges.append(rel)

        if isinstance(body.func, ast.Name):
            if body.func.id in self.ctx:
                node = self.ctx[body.func.id]
            else:
                node = Node.from_ast_body(
                    self.ctx.id,
                    body.func,
                    self.ctx.depth,
                    NodeAttributes(code_name=body.func.id),
                )
                self.ctx[body.func.id] = node
        elif isinstance(body.func, KeyValue):
            if (
                hasattr(body.func, __class_name__)
                and isinstance(body.func.value, ast.Name)
                and body.func.value.id == "self"
            ):
                body.func.value.id = body.func.__class_name__
            if body.func.__str__() in self.ctx:
                node = self.ctx[body.func.__str__()]
            else:
                node = Node.from_ast_body(self.ctx.id, body.func, self.ctx.depth)
                self.ctx[body.func.__str__()] = node
        else:
            node = Node.from_ast_body(self.ctx.id, body.func, self.ctx.depth)

        return self.post_hook(body, node, edges)

    def visit_Import(self, body: ast.Import) -> typing.Any:
        return self.post_hook(
            body,
            node=Node.from_ast_body(self.ctx.id, body, self.ctx.depth),
            edges=[Edge(source_right=c, relation="imports") for c in body.names],
        )

    def visit_ExceptHandler(self, body: ast.ExceptHandler) -> typing.Any:
        if body.type is not None:
            edges = [Edge(source_right=body.type, relation="catches")]
        else:
            edges = []
        edges.extend([Edge(source_right=c, relation="body") for c in body.body])

        attr = None
        if body.name is not None:
            attr = NodeAttributes(code_name=body.name)

        return self.post_hook(
            body,
            node=Node.from_ast_body(self.ctx.id, body, self.ctx.depth, attr),
            edges=edges,
        )

    def visit_Try(self, body: ast.Try) -> typing.Any:
        edges = [Edge(source_right=b, relation="body") for b in body.body]
        edges.extend([Edge(source_right=b, relation="except") for b in body.handlers])
        edges.extend([Edge(source_right=b, relation="finally") for b in body.finalbody])
        edges.extend([Edge(source_right=b, relation="else") for b in body.orelse])

        return self.post_hook(
            body,
            node=Node.from_ast_body(self.ctx.id, body, self.ctx.depth),
            edges=edges,
        )

    def visit_ImportFrom(self, body: ast.ImportFrom) -> typing.Any:
        edges = [Edge(source_right=wrap(body.module), relation="from")]

        edges.extend([Edge(source_right=c, relation="imports") for c in body.names])

        return self.post_hook(
            body,
            node=Node.from_ast_body(self.ctx.id, body, self.ctx.depth),
            edges=edges,
        )

    def visit_alias(self, body: ast.alias) -> typing.Any:
        return self.post_hook(
            body,
            node=Node.from_ast_body(
                self.ctx.id,
                body,
                self.ctx.depth,
                NodeAttributes(
                    import_as=body.asname if body.asname is not None else empty_token,
                    import_name=body.name,
                ),
            ),
        )

    def visit_Assign(self, body: ast.Assign) -> typing.Any:
        edges = []

        for i, target in enumerate(body.targets):
            if isinstance(target, KeyValue) and hasattr(body, __class_name__):
                target.__class_name__ = body.__class_name__
                target.value.id = f"{body.__class_name__}.self"
            edges.append(
                Edge(
                    source_right=target,
                    relation="assigns",
                )
            )
        if isinstance(body.value, ast.Name):
            if hasattr(body, __fn_name__):
                body.value.id = f"{body.__fn_name__}.{body.value.id}"
            if body.value.id in self.ctx:
                node = self.ctx[body.value.id]
            else:
                node = Node.from_ast_body(
                    self.ctx.id,
                    body.value,
                    self.ctx.depth,
                    NodeAttributes(code_name=body.value.id),
                )
                self.ctx[body.value.id] = node
        elif isinstance(body.value, KeyValue):
            if body.value.__str__() not in self.ctx:
                node = self.ctx[body.value.__str__()] = Node.from_ast_body(
                    self.ctx.id, body.value, self.ctx.depth
                )
            else:
                node = self.ctx[body.value.__str__()]
        else:
            node = Node.from_ast_body(self.ctx.id, body.value, self.ctx.depth)

        return self.post_hook(body, node, edges=edges)

    def visit_List(self, node: ast.List) -> typing.Any:
        return self.post_hook(
            node,
            node=Node.from_ast_body(self.ctx.id, node, self.ctx.depth),
            edges=[Edge(source_right=c, relation="list") for c in node.elts],
        )

    def visit_comprehension(self, node: ast.comprehension) -> typing.Any:
        edges = []

        if node.target is not None:
            edges.append(Edge(source_right=node.target, relation="target"))
        if node.iter is not None:
            edges.append(Edge(source_right=node.iter, relation="iter"))
        if node.ifs is not None:
            edges.extend([Edge(source_right=c, relation="ifs") for c in node.ifs])

        return self.post_hook(
            node,
            node=Node.from_ast_body(self.ctx.id, node, self.ctx.depth),
            edges=edges,
        )

    def visit_AugAssign(self, body: ast.AugAssign) -> typing.Any:
        edges = []
        target = body.target

        if isinstance(target, KeyValue) and hasattr(body, __class_name__):
            target.__class_name__ = body.__class_name__
            target.value.id = f"{body.__class_name__}.self"
        edges.append(
            Edge(
                source_right=target,
                relation=body.op.__class__.__name__,
            )
        )
        del body.op
        if isinstance(body.value, ast.Name):
            if hasattr(body, __fn_name__):
                body.value.id = f"{body.__fn_name__}.{body.value.id}"
            if body.value.id in self.ctx:
                node = self.ctx[body.value.id]
            else:
                node = Node.from_ast_body(
                    self.ctx.id,
                    body.value,
                    self.ctx.depth,
                    NodeAttributes(code_name=body.value.id),
                )
                self.ctx[body.value.id] = node
        elif isinstance(body.value, KeyValue):
            if body.value.__str__() not in self.ctx:
                node = self.ctx[body.value.__str__()] = Node.from_ast_body(
                    self.ctx.id, body.value, self.ctx.depth
                )
            else:
                node = self.ctx[body.value.__str__()]
        else:
            node = Node.from_ast_body(self.ctx.id, body.value, self.ctx.depth)

        return self.post_hook(body, node, edges=edges)

    def visit_arg(self, body: ast.arg) -> typing.Any:
        edges = []

        if body.arg in self.ctx:
            node = self.ctx[body.arg]
        else:
            node = Node.from_ast_body(
                self.ctx.id, body, self.ctx.depth, NodeAttributes(code_name=body.arg)
            )
            self.ctx[body.arg] = node

        if body.annotation is not None:
            node.attr.type_name = parse_type(body.annotation)
            del body.annotation

        if hasattr(body, "default"):
            edges.append(Edge(source_right=body.default, relation="default"))

        return self.post_hook(body, node, edges)

    def visit_arguments(self, node: ast.arguments) -> typing.Any:
        for i, default in enumerate(node.defaults):
            setattr(node.args[i], "default", default)

        return self.post_hook(node, None)

    def visit_FunctionDef(self, body: ast.FunctionDef) -> typing.Any:
        node = Node.from_ast_body(
            self.ctx.id, body, self.ctx.depth, NodeAttributes(code_name=body.name)
        )

        if body.returns is not None:
            node.attr.return_type = parse_type(body.returns)
        docstring = ast.get_docstring(body)
        if docstring is not None:
            node.attr.docstring = docstring

        edges = []
        i = 0
        if body.args.args is not None:
            for arg in body.args.args:
                edges.append(
                    Edge(source_right=arg, relation="named_arg", attr=EdgeAttributes())
                )
                i += 1
        if body.args.vararg is not None:
            edges.append(
                Edge(
                    source_right=body.args.vararg,
                    relation="vararg",
                    attr=EdgeAttributes(),
                )
            )
            i += 1
        if body.args.kwarg is not None:
            edges.append(
                Edge(
                    source_right=body.args.kwarg,
                    relation="kwarg",
                    attr=EdgeAttributes(),
                )
            )
        for i, child in enumerate(body.body):
            if isinstance(child, ast.Return):
                edges.append(Edge(source_right=wrap(child.value), relation="returns"))
            else:
                edges.append(
                    Edge(source_right=child, relation="body", attr=EdgeAttributes())
                )

        if node is not None:
            self.ctx[body.name] = node

        return self.post_hook(body, node, edges)

    def visit_Return(self, body: ast.Return) -> typing.Any:
        return self.generic_visit(body)

    def visit_Name(self, body: ast.Name) -> typing.Any:
        if body.id in self.ctx:
            return self.post_hook(body, self.ctx[body.id])
        node = Node.from_ast_body(
            self.ctx.id, body, self.ctx.depth, NodeAttributes(code_name=body.id)
        )
        self.ctx[body.id] = node

        return self.post_hook(body, node)

    def visit_UnaryOp(self, body: ast.UnaryOp) -> typing.Any:
        return self.post_hook(
            body.operand,
            node=Node.from_ast_body(self.ctx.id, body.operand, self.ctx.depth),
        )

    def visit_BinOp(self, body: ast.BinOp) -> typing.Any:
        edges = [Edge(source_right=body.right, relation=body.op.__class__.__name__)]

        del body.op

        return self.post_hook(
            body,
            node=Node.from_ast_body(self.ctx.id, body.left, self.ctx.depth),
            edges=edges,
        )


    def visit_BoolOp(self, body: ast.BoolOp) -> typing.Any:
        edges = [
            Edge(source_right=v, relation=body.op.__class__.__name__)
            for v in body.values
        ]

        del body.op

        return self.post_hook(
            body,
            node=Node.from_ast_body(self.ctx.id, body, self.ctx.depth),
            edges=edges,
        )

    def visit_Expr(self, node: ast.Expr) -> typing.Any:
        return self.post_hook(
            wrap(node.value),
            node=Node.from_ast_body(self.ctx.id, wrap(node.value), self.ctx.depth),
        )

    def visit_Dict(self, body: ast.Dict) -> typing.Any:
        edges = []

        for i, value in enumerate(body.values):
            key = body.keys[i]
            value = KeyValue(wrap(key), wrap(value))
            body.values[i] = value
            edges.append(Edge(source_right=value, relation="value"))

        del body.keys

        return self.post_hook(
            body,
            node=Node.from_ast_body(self.ctx.id, body, self.ctx.depth),
            edges=edges,
        )

    def visit_Subscript(self, node: ast.Subscript) -> typing.Any:
        edges = [Edge(source_right=node.slice, relation="slice")]
        return self.post_hook(
            node,
            node=Node.from_ast_body(self.ctx.id, node.value, self.ctx.depth),
            edges=edges,
        )

    def visit_KeyValue(self, body: KeyValue) -> typing.Any:
        if hasattr(body, __class_name__):
            setattr(body.key, __class_name__, body.__class_name__)
            setattr(body.value, __class_name__, body.__class_name__)
            if isinstance(body.value, ast.Name) and body.value.id == "self":
                body.value.id = f"{body.__class_name__}.self"
        if hasattr(body, __fn_name__):
            setattr(body.key, __fn_name__, body.__fn_name__)
            setattr(body.value, __fn_name__, body.__fn_name__)
            if (
                isinstance(body.value, ast.Name)
                and f"{body.__fn_name__}.{body.value.id}" in self.ctx
            ):
                body.value.id = f"{body.__fn_name__}.{body.value.id}"

        if not body.__str__() in self.ctx:
            node = self.ctx[body.__str__()] = Node.from_ast_body(
                self.ctx.id, body, self.ctx.depth
            )
            edges = [
                Edge(source_right=body.key, relation="key"),
                Edge(source_right=body.value, relation="value"),
            ]
        else:
            node = self.ctx[body.__str__()]
            del body.key
            edges = None

        return self.post_hook(
            body,
            node=node,
            edges=edges,
        )

    def post_hook(
        self,
        body: ast.AST,
        node: typing.Optional["Node"],
        edges: typing.Iterable[Edge] = None,
    ) -> typing.Any:
        if hasattr(body, __edges__) and node is not None:
            for edge in body.__edges__:
                if edge.source_right == body and edge not in self.ctx.edges:
                    edge.node_right = node
                    self.ctx.edges.append(edge)
        if edges is not None and node is not None:
            for edge in edges:
                edge.source_left = body
                edge.node_left = node
                if hasattr(edge.source_right, __edges__):
                    edge.source_right.__edges__.append(edge)
                else:
                    setattr(edge.source_right, __edges__, [edge])

        if node is not None:
            self.ctx.add_node(node)

        return self.generic_visit(body)

    def generic_visit(self, node):
        self.ctx.depth += 1

        for ast_field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast.AST):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, ast.AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, ast_field)
                else:
                    setattr(node, ast_field, new_node)
        return node

    def simple_visit(self, node):
        return self.post_hook(
            body=node,
            node=Node.from_ast_body(self.ctx.id, node, self.ctx.depth),
        )
