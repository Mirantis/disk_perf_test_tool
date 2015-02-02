#!/bin/bash

port_ids=$(neutron port-list | tail -n+4 | head -n-1 | awk '{print $2}')

for port_id in $port_ids; do
	neutron port-delete $port_id
done

subnet_list=$(neutron subnet-list | grep rally | awk '{print $2}')

for subnet_id in $subnet_id; do
	neutron subnet-delete $subnet_id
done

net_ids=$(neutron net-list | grep rally | awk '{print $2}')

for net_id in $net_ids; do
	neutron net-delete $net_id
done
