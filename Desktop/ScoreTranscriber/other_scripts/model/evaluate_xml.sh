#!/usr/bin/env bash
#export XDG_RUNTIME_DIR=/media/hdd1/data/zx_ofyp/model/
export QT_QPA_PLATFORM=offscreen

if [ "$#" -ne 3 ]; then
    echo "USAGE: evaluate.bash ref.xml transcription.xml MV2H_path"
    exit 1
fi

musescore3 -o $1.mid $1
musescore3 -o $2.mid $2

java -Xdiag -Xms2048m -Xmx2048m -cp $3 mv2h.tools.Converter -i $1.mid >$1.conv.txt
java -Xdiag -Xms2048m -Xmx2048m -cp $3 mv2h.tools.Converter -i $2.mid >$2.conv.txt
rm $1.mid $2.mid

java -Xdiag -Xms2048m -Xmx2048m -cp $3 mv2h.Main -g $1.conv.txt -t $2.conv.txt -a
rm $1.conv.txt $2.conv.txt