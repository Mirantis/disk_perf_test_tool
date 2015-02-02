import gspread
from config import EMAIL, PASSWORD, DOCUMENT_ID, WORK_SHEET
from make_report import get_work_sheet, append_row


class Measurement(object):
    def __init__(self):
        self.build = ""
        self.build_type = 0  # GA/Master/Other
        self.md5 = ""
        self.results = {
            "": (float, float)
        }


class Storage(object):

    def __init__(self, email, password, doc_id, work_sheet_name):
        self.gc = gspread.login(email, password)
        self.sh = self.gc.open_by_key(doc_id)
        self.work_sheet = get_work_sheet(self.sh, work_sheet_name, 40)

    def store(self, data):
        append_row(self.work_sheet, data)

    def retrieve(self, id):
        row_number = self.find_by_id(id)

        if row_number != -1:
            vals = self.work_sheet.row_values(row_number)
            m = Measurement()
            m.build = vals["build_id"]
            del vals["build_id"]
            m.build_type = vals["type"]
            del vals["type"]
            m.md5 = vals["iso_md5"]
            del vals["iso_md5"]
            m.results = {k: vals[k] for k in vals.keys()}
        else:
            return None

    def find_by_id(self, row_id):
        for i in range(1, self.work_sheet):
            if self.work_sheet.cell(i, 1) == row_id:
                return i

        return -1


