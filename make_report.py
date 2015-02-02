import gspread
import argparse
import json

from config import DEFAULT_FILE_PATH, PASSWORD, EMAIL, \
    WORK_SHEET, DOCUMENT_ID, ROW_COUNT
from gspread.exceptions import WorksheetNotFound


def load_data(file_name):
    with open(file_name) as f:
        data = f.read()
        return json.loads(data)


#getting worksheet from sheet or create it with specified column names.
def get_work_sheet(sheet, name, column_names):
    try:
        work_sheet = sheet.worksheet(name)
    except WorksheetNotFound:
        work_sheet = sheet.add_worksheet(title=name, rows=ROW_COUNT,
                                         cols=max(40, len(column_names)))

        for i in range(1, len(column_names) + 1):
            work_sheet.update_cell(1, i, column_names[i - 1])

    return work_sheet


def get_row_number(work_sheet):
    num = 2

    while num < work_sheet.row_count and work_sheet.cell(num, 1).value != "":
        num += 1

    if num == work_sheet.row_count:
        work_sheet.append_row(["" for x in range(work_sheet.col_count)])

    return num


def append_row(work_sheet, row):
    row_number = get_row_number(work_sheet)

    i = 1
    for k in row.keys():
        work_sheet.update_cell(row_number, i, row[k])
        i += 1


def make_report(email, password, data):
    gc = gspread.login(email, password)
    sh = gc.open_by_key(DOCUMENT_ID)

    work_sheet = get_work_sheet(sh, WORK_SHEET, data.keys())
    append_row(work_sheet, data)


def main(file_name, email, password):
    data = load_data(file_name)
    make_report(email, password, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help='data file path',
                        default=DEFAULT_FILE_PATH)
    parser.add_argument('-e', '--email', help='user email',
                        default="aaa@gmail.com")
    parser.add_argument('-p', '--password', help='user password',
                        default="1234")
    parser.add_argument('-m', '--mode', help='mode type local or global',
                        default=DEFAULT_FILE_PATH)
    results = parser.parse_args()
    print results
    # main(file_name, email, password)

