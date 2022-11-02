import ast


class ClassDef(ast.stmt):
    def __init__(
        self,
        identifier_name,
        bases,
        keywords,
        cls,
        slf,
        properties,
        body,
        decorator_list,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cls = cls
        self.slf = slf
        self.properties = properties
        self.name = identifier_name
        self.bases = bases
        self.keywords = keywords
        self.body = body
        self.decorator_list = decorator_list

    _fields = (
        "name",
        "bases",
        "keywords",
        "cls",
        "slf",
        "properties",
        "body",
        "decorator_list",
    )
