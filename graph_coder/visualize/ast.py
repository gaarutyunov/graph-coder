#  Copyright 2023 German Arutyunov
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from io import BytesIO, FileIO
from pathlib import Path
from typing import Tuple

import black
import matplotlib.pyplot as plt
import networkx as nx
from black import Mode
from matplotlib import font_manager
from networkx.drawing.nx_agraph import graphviz_layout
from PIL import Image, ImageDraw, ImageFont

font_path = Path(__file__).parent.parent.parent / "assets/Space_Mono"

font_files = font_manager.findSystemFonts(fontpaths=[font_path], fontext="ttf")

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)


def export_graph(graph: nx.Graph, out: FileIO, mode: str = "L"):
    """Export graph to image."""
    pos = graphviz_layout(graph, prog="dot")

    nx.draw(
        graph, pos=pos, with_labels=True, node_color="#ffffff", font_family="Space Mono"
    )
    nx.draw_networkx_edge_labels(
        graph,
        pos=pos,
        font_size=8,
        font_color="#000000",
        font_family="Space Mono",
        edge_labels=nx.get_edge_attributes(graph, "label"),
    )

    buf = BytesIO()

    plt.savefig(buf, format="tiff", pil_kwargs={"format": "TIFF"})

    im = Image.open(buf)
    im = im.convert(mode)
    im.save(out, format="TIFF")


def export_tokens(
    graph: nx.Graph,
    out: FileIO,
    node_num: int = 5,
    edge_num: int = 3,
    image_size: Tuple[int, int] = (544, 256),
    font_size: int = 13,
    mode: str = "L",
    color: int = 0,
):
    """Export tokens to image."""
    texts = []

    for i, label in enumerate(graph.nodes):
        if i == node_num:
            break
        texts.append(label)

    for i, (_, _, label) in enumerate(graph.edges(data="label")):
        if i == edge_num:
            break
        texts.append(label)
    font = ImageFont.truetype(str(font_path / "SpaceMono-Regular.ttf"), size=font_size)

    with Image.new(mode, size=image_size, color="white") as f:
        d = ImageDraw.Draw(f, mode=mode)

        offset = 10
        for i, text in enumerate(texts):
            if i == node_num:
                d.text((10, offset), text="...", font=font, fill=color)
                offset += font_size + 4
            d.text((10, offset), text=text, font=font, fill=color)
            offset += font_size + 4
            if i == len(texts) - 1:
                d.text((10, offset), text="...", font=font, fill=color)
                offset += font_size + 4

        font_big = ImageFont.truetype(
            str(font_path / "SpaceMono-Regular.ttf"), size=font_size * 2
        )
        g = nx.convert_node_labels_to_integers(graph, first_label=0, ordering="sorted")

        offset_y, offset_x = offset + font_size, 10
        d.text((offset_x, offset_y - 2), text="(", font=font_big, fill=color)
        offset_x += font_size + 8
        for i, (u, v) in enumerate(g.edges):
            if i == edge_num * 2:
                break
            if i == edge_num:
                d.text((offset_x - 4, offset_y), text="...", font=font, fill=color)
                d.text(
                    (offset_x - 4, offset_y + font_size + 4),
                    text="...",
                    font=font,
                    fill=color,
                )
                offset_x += font_size * 2 + 4
            d.text((offset_x, offset_y), text=str(u), font=font, fill=color)
            d.text(
                (offset_x, offset_y + font_size + 4), text=str(v), font=font, fill=color
            )
            offset_x += font_size * 2
        d.text((offset_x, offset_y - 2), text=")", font=font_big, fill=color)

        f.save(out, format="TIFF")


def export_code(
    code: str,
    out: FileIO,
    image_size: Tuple[int, int] = (480, 352),
    font_size: int = 16,
    xy: Tuple[int, int] = (10, 10),
    line_length: int = 25,
    mode: str = "L",
):
    """Export code to image."""
    code = black.format_str(code, mode=Mode(line_length=line_length))
    font = ImageFont.truetype(str(font_path / "SpaceMono-Regular.ttf"), size=font_size)

    with Image.new(mode, size=image_size, color="white") as f:
        d = ImageDraw.Draw(f, mode=mode)
        d.text(xy, text=code, font=font, fill=0)

        f.save(out, format="TIFF")
