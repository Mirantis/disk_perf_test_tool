.PHONY: mypy


ALL_FILES=$(shell find wally/ -type f -name '*.py')
STUBS="stubs:.env/lib/python3.5/site-packages"

mypy:
		bash -c "cd ~/workspace/wally; source .env/bin/activate ; MYPYPATH=${STUBS} python3 -m mypy -s ${ALL_FILES}"
