FLAVOR_NAME="disk_io_perf.1024"

SERV_GROUPS="disk_io_perf.aa.0 disk_io_perf.aa.1 disk_io_perf.aa.2 disk_io_perf.aa.3 disk_io_perf.aa.4 disk_io_perf.aa.5 disk_io_perf.aa.6 disk_io_perf.aa.7"

KEYPAIR_NAME="disk_io_perf"
IMAGE_NAME="disk_io_perf"
KEY_FILE_NAME="${KEYPAIR_NAME}.pem"
IMAGE_URL="https://cloud-images.ubuntu.com/trusty/current/trusty-server-cloudimg-amd64-disk1.img"
IMAGE_USER="ubuntu"
NETWORK_ZONE_NAME="net04"
FL_NETWORK_ZONE_NAME="net04_ext"
VM_COUNT="x1"
TESTER_TYPE="iozone"
RUNNER="ssh"
SECGROUP='disk_io_perf'
