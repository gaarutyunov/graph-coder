from graph_coder.data.features import replace_special_tokens


def test_special_tokens():
    src = "def f(x):\n\tif x > 0:\n\t\tx = 1\n\telse:\n\t\tx = 2\n\treturn x\n"

    src = replace_special_tokens(src)

    assert src == "def f(x):[NET]if x > 0:[NET][TAB]x = 1[NET]else:[NET][TAB]x = 2[NET]return x[NEW]"

    src = replace_special_tokens(src)

    assert src == "def f(x):\n\tif x > 0:\n\t\tx = 1\n\telse:\n\t\tx = 2\n\treturn x\n"
