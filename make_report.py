import gspread
import argparse
import json

from config import DEFAULT_FILE_PATH, \
    WORK_SHEET, DOCUMENT_ID, ROW_COUNT, OUTPUT_FILE
from storage_api import DiskStorage, GoogleDocsStorage, get_work_sheet, append_row


def load_data(file_name):
    with open(file_name) as f:
        data = f.read()
        return json.loads(data)


#getting worksheet from sheet or create it with specified column names.

def make_report(email, password, data):
    gc = gspread.login(email, password)
    sh = gc.open_by_key(DOCUMENT_ID)

    work_sheet = get_work_sheet(sh, WORK_SHEET, data.keys())
    append_row(work_sheet, data)


def main(file_name, email, password, mode):
    data = load_data(file_name)

    if mode == 'local':
        storage = DiskStorage(OUTPUT_FILE)
    else:
        storage = GoogleDocsStorage(DOCUMENT_ID, WORK_SHEET, email, password)

    storage.store(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help='data file path',
                        default=DEFAULT_FILE_PATH)
    parser.add_argument('-e', '--email', help='user email',
                        default="aaa@gmail.com")
    parser.add_argument('-p', '--password', help='user password',
                        default="1234")
    parser.add_argument('-m', '--mode', help='mode type local or global',
                        default='local')
    results = parser.parse_args()
    main(results.name, results.email, results.password, results.mode)

