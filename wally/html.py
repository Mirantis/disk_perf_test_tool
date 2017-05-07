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


def table(caption: str, headers: Optional[List[str]], data: List[List[str]], align: List[str] = None) -> str:
    doc = xmlbuilder3.XMLBuilder("table",
                                 **{"class": "table table-bordered table-striped table-condensed table-hover",
                                    "style": "width: auto;"})

    doc.caption.H3.center(caption)

    if headers is not None:
        with doc.thead:
            with doc.tr:
                for header in headers:
                    doc.th(header)

    max_cols = max(len(line) for line in data if not isinstance(line, str))

    with doc.tbody:
        for line in data:
            with doc.tr:
                if isinstance(line, str):
                    with doc.td(colspan=str(max_cols)):
                        doc.center.b(line)
                else:
                    if align:
                        for vl, col_align in zip(line, align):
                            doc.td(vl, align=col_align)
                    else:
                        for vl in line:
                            doc.td(vl)

    return xmlbuilder3.tostr(doc).split("\n", 1)[1]