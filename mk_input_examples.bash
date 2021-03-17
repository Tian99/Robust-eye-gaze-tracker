#!/usr/bin/env bash
set -euo pipefail
trap 'e=$?; [ $e -ne 0 ] && echo "$0 exited in error"' EXIT
env|grep -q ^DRYRUN=.&&DRYRUN=echo||DRYRUN=""

REDO="1" # set to non-empty to redo ffmpage 

# 20210228WF - make samples

#vcodec="copy.mov" # keep mov container. doesn't work for shorter segments?
vcodec=h264.mp4    # get away from quicktime. 264 w/mp4 segments work at any length

while read f desc t_start t_end; do
   out="input/examples/${desc}_$t_start-${t_end}_$(basename $f .mov).${vcodec#*.}"
   echo "# $f - $out"
   [ -r "$out" -a -z "$REDO" ] && continue
   test -d $(dirname "$out") || mkdir $_
   # N.B. ffmpeg eats input. feed it dev/null so it doesn't consumpe 'i' in input of next command
   $DRYRUN ffmpeg -y -i "$f" -ss $t_start -to $t_end -c:v ${vcodec%.*} "$out" -nostdin
done <<HEREDOC
input/11710_20200911_run1.mov right  0:0:54   0:0:54.5
input/11710_20200911_run1.mov left   0:0:59.8 0:1:00.1
input/11710_20200911_run1.mov blink  0:2:05.5 0:2:06.5
HEREDOC

$DRYRUN ffmpeg -i input/11710_20200911_run1.mov -vf "select=eq(n\,34)" -vframes 1 input/examples/11710_run1_frame34.png
