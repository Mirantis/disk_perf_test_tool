Installation
------------

Only Redhat/Ubuntu/Debian distros are supported

	# git clone https://github.com/Mirantis/disk_perf_test_tool wally
	# cd wally

	# git clone https://github.com/Mirantis/disk_perf_test_tool.git wally
	# cd wally

For small installation (test inly)

	# ./install.sh

For full installation (test + html report)

	# ./insall.sh --full

Manual installation:

Install : pip, python-openssl python-novaclient python-cinderclient
python-keystoneclient python-glanceclient python-faulthandler,
python-scipy python-numpy python-matplotlib python-psutil

Then run
	
	# pip install -r requirements.txt

Create a directory for configs and copy wally/config_examples/default.yaml
in it.

Create a directory for results and update default.yaml
settings::results_storage value to point to this directory.

Copy appropriate file from wally/config_examples into the same folder,
where default.yaml stored, update it, accordingly to you system and run
wally

$ export PYTHONPATH="$PYTHONPATH:WALLY_DIR"

for python 2.7

$ python -m wally test "my test comment" CONFIG_FILE

for python 2.6 or 2.7

$ python -m wally.\_\_main\_\_ test "my test comment" CONFIG_FILE
