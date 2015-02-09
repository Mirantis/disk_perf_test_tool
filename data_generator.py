import os
import sys
import uuid
import random
import itertools

from petname import Generate as pet_generate
from storage_api import create_storage

from report import ssize_to_kb

types = ["GA", "master"] + [pet_generate(2, '-') for _ in range(2)]
random.shuffle(types)
tp = itertools.cycle(types)

sz = ["1k", "4k", "64k", "256k", "1m"]
op_type = ["randread", "read", "randwrite", "write"]
is_sync = ["s", "a"]

storage = create_storage(sys.argv[1], "", "")
combinations = list(itertools.product(op_type, is_sync, sz))

for i in range(30):
    row = {"build_id": pet_generate(2, " "),
           "type": next(tp),
           "iso_md5": uuid.uuid4().get_hex()}

    for op_type, is_sync, sz in combinations:
        ((random.random() - 0.5) * 0.2 + 1)
        row[" ".join([op_type, is_sync, sz])] = (
            ((random.random() - 0.5) * 0.2 + 1) * (ssize_to_kb(sz) ** 0.5),
        ((random.random() - 0.5) * 0.2 + 1) * (ssize_to_kb(sz) ** 0.5) * 0.15)

    print len(row)
    storage.store(row)
