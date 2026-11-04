#!/bin/bash
# Delete nipype's crash files ("crash-*.pklz"). The workflow keeps them in
# ./output/_work/crash (it used to write them into ./src).
find ./output/_work ./src -name "crash-*.pklz" -delete 2>/dev/null
exit 0
