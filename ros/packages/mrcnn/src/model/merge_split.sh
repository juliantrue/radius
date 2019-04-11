#!/bin/sh

FILE_NAME=$1
if [ -z "${FILE_NAME}" ]; then
  echo "Required to pass file name of file to be reconstructed or split."
  echo "Usage: merge_split.sh mask_rcnn_coco.h5"
  exit 1
fi

if [ ! -f "${FILE_NAME}" ]; then
  echo "Merging multiple files..."
  cat ${FILE_NAME}.* > ${FILE_NAME}
  rm ${FILE_NAME}.*
else
  echo "Splitting up file over 50Mb"
  split -b 50m -d ${FILE_NAME} ${FILE_NAME}.
  rm ${FILE_NAME}
fi
