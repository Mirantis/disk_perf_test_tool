import sys
import uuid
import random
import itertools

from petname import Generate as pet_generate
from storage_api import create_storage


types = ["GA", "master"] + [pet_generate(2, '-') for _ in range(2)]
random.shuffle(types)
tp = itertools.cycle(types)

sz = ["4k", "64k", "1m"]
op_type = ["randread", "read", "randwrite", "write"]
is_sync = ["s", "a"]

storage = create_storage("file:///home/gstepanov/rally-results-processor/sample.json", "", "")
combinations = list(itertools.product(op_type, is_sync, sz))

for i in range(30):
    row = {"build_id": pet_generate(2, " "),
           "type": next(tp),
           "iso_md5": uuid.uuid4().get_hex()}

    for sz, op_type, is_sync in combinations:
        row[" ".join([sz, op_type, is_sync])] = (random.random() * 100,
                                                 random.random() * 5)

    print row
    storage.store(row)
