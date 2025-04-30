#echo "${SCRIPT_PATHS[@]}"
SCRIPT_PATHS=("$@")

#echo "Array content: ${SCRIPT_PATHS[@]}"
for script in "${SCRIPT_PATHS[@]}"; do
    echo " $script"
    pkill -9 -f python
    sudo rm /tmp/libtpu_lockfile
    source ~/miniconda3/bin/activate base;

done
