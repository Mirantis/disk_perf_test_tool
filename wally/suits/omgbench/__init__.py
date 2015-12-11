import os.path


import texttable


from ..itest import TwoScriptTest


class OmgTest(TwoScriptTest):
    root = os.path.dirname(__file__)
    pre_run_script = os.path.join(root, "prepare.sh")
    run_script = os.path.join(root, "run.sh")

    @classmethod
    def format_for_console(cls, data):
        results = []

        for res in data[0]:
            results.append(float(res.raw_result))

        totalt = sum(results)
        totalms = int(100 * 2 * len(results) / totalt)
        tab = texttable.Texttable(max_width=120)
        tab.set_deco(tab.HEADER | tab.VLINES | tab.BORDER)
        tab.header(["Bandwidth total"])
        tab.add_row([totalms])
        return tab.draw()
