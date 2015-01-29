import gspread
import argparse
import json

from config import DEFAULT_FILE_PATH, PASSWORD, EMAIL, COL_COUNT, WORK_SHEET, DOCUMENT_ID


def load_data(file_name):
    with open(file_name) as f:
        data = f.read()
        return json.loads(data)


def make_report(data):
    gc = gspread.login(EMAIL, PASSWORD)
    sh = gc.open_by_key(DOCUMENT_ID)
    worksheet = sh.add_worksheet(title=WORK_SHEET, rows=len(data.keys()), cols=COL_COUNT)

    i = 1
    for k in data.keys():
        worksheet.update_cell(i, 1, k)
        worksheet.update_cell(i, 2, data[k])
        i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help='data file path', default=DEFAULT_FILE_PATH)
    results = parser.parse_args()
    data = load_data(results.name)
    make_report(data)
