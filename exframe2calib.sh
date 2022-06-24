#!/bin/bash

# Paulo Santiago e Allan S. Pinto - 2021

# PATH=calibvid
WORKING_DIR=$(pwd)

VIDEO_CALIB_DIR=$WORKING_DIR/calibvid
# echo $VIDEO_CALIB_DIR

if [ -e "snap_calib" ]; then
    echo "WARNING!: folder 'snap_calib' already exist!"
    exit
fi

mkdir $WORKING_DIR/snap_calib/
DIR_PICTURES=~/Pictures/
FILES_VLC=vlcsnap-*

COUNTER=1
for file in $(ls ${VIDEO_CALIB_DIR}/); do
    clear
    CAMFOLDER=c$COUNTER

#    if [ -e $(ls "$DIR_PICTURES$FILES_VLC") ]; then
        echo "WARNING!: If files vlcsnap-* exist in directory '$DIR_PICTURES' wil be moved!"
        BACKUP_FOLDER=$(date +%d%h%s)
        mkdir ~/Pictures/backup_vlcsnap_$BACKUP_FOLDER
        mv -f ~/Pictures/vlcsnap-* ~/Pictures/backup_vlcsnap_$BACKUP_FOLDER
#    fi

    SNAP_FOLDER=$WORKING_DIR/snap_calib/c$COUNTER
    mkdir $SNAP_FOLDER
    read -p "Press [Enter] key to open VLC Player"
    vlc $VIDEO_CALIB_DIR/${file} &
    printf "Videos: ${file}\n"
    printf "snapshot frames to calibration, press Shift + s\n"
    read -p "Press [Enter] key to open next video"
#    echo $SNAP_FOLDER
#    read -p ""
#    echo $SNAP_FOLDER
#    read -p 
    mv -f ~/Pictures/vlcsnap-* $SNAP_FOLDER
#    clear
    COUNTER=$((COUNTER+1))
#    echo $COUNTER
done

clear
printf "\nSnapshots have been saved in directory: snapsho_calib\n\n"

exit 0

