from typing import Optional, List, Callable


import xmlbuilder3


eol = "<br>"


def tag(name: str) -> Callable[[str], str]:
    def closure(data: str) -> str:
        return "<{}>{}</{}>".format(name, data, name)
    return closure


H3 = tag("H3")
H2 = tag("H2")
center = tag("center")


def img(link: str) -> str:
    return '<img src="{}">'.format(link)


def table(caption: str, headers: Optional[List[str]], data: List[List[str]]) -> str:
    doc = xmlbuilder3.XMLBuilder("table",
                                 **{"class": "table table-bordered table-striped table-condensed table-hover",
                                    "style": "width: auto;"})

    doc.caption.H3.center(caption)

    if headers is not None:
        with doc.thead:
            with doc.tr:
                for header in headers:
                    doc.th(header)

    with doc.tbody:
        for line in data:
            with doc.tr:
                for vl in line:
                    doc.td(vl)

    return xmlbuilder3.tostr(doc).split("\n", 1)[1]