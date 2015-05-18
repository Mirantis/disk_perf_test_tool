import os.path


import texttable


from ..itest import TwoScriptTest


class PgBenchTest(TwoScriptTest):
    root = os.path.dirname(__file__)
    pre_run_script = os.path.join(root, "prepare.sh")
    run_script = os.path.join(root, "run.sh")

    @classmethod
    def format_for_console(cls, data):
        tab = texttable.Texttable(max_width=120)
        tab.set_deco(tab.HEADER | tab.VLINES | tab.BORDER)
        tab.header(["TpmC"])
        tab.add_row([data['res']['TpmC']])
        return tab.draw()
