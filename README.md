Block storage devices tests tool. See wiki for details
Look into INSTALL.md for installation steps.

Look into config-example for examples of config file.
Copy example in same folder and replace ${VAR} with appropriate value



|          | Size, TiB | $/TiB | IOPS WR | IOPS RD | BW WR MB| BW RD MB| Lat ms |
|:--------:|----------:|:-----:|:-------:|:-------:|:-------:|:-------:|:------:|
| SATA HDD | 10        | 25-50 | 50-150  | 50-150  | 100-200 | 100-200 | 3-7    |
| SSD      | 2         | 200   | 100-5k  | 1k-20k  |  50-400 | 200-500 | 0.1-1  |
| NVME     | 2         | 400   | 400-20k | 2k-50k  | 200-1.5k| 500-2k  | 0.01-1 |