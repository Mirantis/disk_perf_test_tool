.PHONY: mypy pylint pylint_e


ALL_FILES=$(shell find wally/ -type f -name '*.py')
STUBS="stubs:.env/lib/python3.5/site-packages"
ACTIVATE=cd ~/workspace/wally; source .env/bin/activate

mypy:
		bash -c "${ACTIVATE}; MYPYPATH=${STUBS} python3 -m mypy -s ${ALL_FILES}"


PYLINT_FMT=--msg-template={path}:{line}: [{msg_id}({symbol}), {obj}] {msg}

pylint:
		bash -c "${ACTIVATE} ; python3 -m pylint '${PYLINT_FMT}' --rcfile=pylint.rc ${ALL_FILES}"

pylint_e:
		bash -c "${ACTIVATE} ; python3 -m pylint -E '${PYLINT_FMT}' --rcfile=pylint.rc ${ALL_FILES}"
