#!/bin/bash
# usage: run_cases.sh OUT.tsv TIMEOUT SUITE CASE... ; runs every case with the three kernels,
# one process each, killed after TIMEOUT seconds (recorded as a TIMEOUT row).
OUT=$1; TO=$2; SUITE=$3; shift 3
D=$(cd "$(dirname "$0")" && pwd)
for c in "$@"; do
  for k in ${KERNELS:-default relk auto}; do
    timeout $TO julia --project=$D $D/bench.jl $SUITE $c $k $OUT
    rc=$?
    [ $rc -eq 124 ] && printf '%s\t%s\t%s\tall\t-\tTIMEOUT>%s\t-\n' $SUITE $c $k $TO >> $OUT
    [ $rc -ne 0 ] && [ $rc -ne 124 ] && printf '%s\t%s\t%s\tall\t-\tERROR(rc=%s)\t-\n' $SUITE $c $k $rc >> $OUT
  done
done
