SSH_PASS=$(sshpass)
export SSH_OPTS="-o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no"

if [ -z SSH_PASS ]; then
     sudo apt-get install sshpass
     echo 'All dependencies has been installed'
fi

DEBS=`download_debs`
echo "Debs has been downloaded"

bash run_test.sh 172.16.52.108  172.16.55.2 disk_io_perf.pem file_to_test.dat result.txt
bash run_test.sh 172.16.52.108  172.16.55.2 disk_io_perf.pem file_to_test.dat result.txt
bash run_test.sh 172.16.52.108  172.16.55.2 disk_io_perf.pem file_to_test.dat result.txt
bash run_test.sh 172.16.52.108  172.16.55.2 disk_io_perf.pem file_to_test.dat result.txt