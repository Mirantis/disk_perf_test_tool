Installation
------------

Only Redhat/Ubuntu/Debian distros supported

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
