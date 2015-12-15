import os.path


import texttable


from ..itest import TwoScriptTest


class OmgTest(TwoScriptTest):
    root = os.path.dirname(__file__)
    pre_run_script = os.path.join(root, "prepare.sh")
    run_script = os.path.join(root, "run.sh")

    @classmethod
    def format_for_console(cls, data):
        success_vals = []
        duration_vals = []
        count = 0
        for res in data[0]:
            msgs, success, duration = res.raw_result.strip().split('\n')
            count += msgs
            success_vals.append(float(success))
            duration_vals.append(float(duration))

        totalt = max(duration_vals)
        totalms = int(count / totalt)
        sucesst = int(sum(success_vals) / len(success_vals))
        tab = texttable.Texttable(max_width=120)
        tab.set_deco(tab.HEADER | tab.VLINES | tab.BORDER)
        tab.header(["Bandwidth m/s", "Success %"])
        tab.add_row([totalms, sucesst])
        return tab.draw()
