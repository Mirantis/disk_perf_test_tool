.PHONY: mypy pylint pylint_e docker

ALL_FILES=$(shell find wally/ -type f -name '*.py')
STUBS="stubs:../venvs/wally/lib/python3.5/site-packages/"

mypy:
		MYPYPATH=${STUBS} python -m mypy --ignore-missing-imports --follow-imports=skip ${ALL_FILES}

PYLINT_FMT=--msg-template={path}:{line}: [{msg_id}({symbol}), {obj}] {msg}

pylint:
		python -m pylint '${PYLINT_FMT}' --rcfile=pylint.rc ${ALL_FILES}

pylint_e:
		python3 -m pylint -E '${PYLINT_FMT}' --rcfile=pylint.rc ${ALL_FILES}

docker:
		docker build --squash -t wally:v2 .
		docker tag wally:v2 ${DOCKER_ID_USER}/wally:v2

docker_push:
		docker push ${DOCKER_ID_USER}/wally:v2

