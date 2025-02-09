#!/bin/bash

TO_SUBMIT_JOBS=$(ls ../configs | grep "inference.yaml")

ALL_TASK=$1
# ALL_TASK=""
# ALL_TASK="${ALL_TASK} MMAU/test"
# ALL_TASK="${ALL_TASK} MusicCaps-AudioCaptioning/test"
# ALL_TASK="${ALL_TASK} MusicCaps-AudioCaptioning/test"
# ALL_TASK="${ALL_TASK} audiocaps-AudioCaptioning/interleaved_knn-test"
# ALL_TASK="${ALL_TASK} MusicCaps-AudioCaptioning/interleaved_knn-test"

# # # ===== Classification =====
# ALL_TASK="${ALL_TASK} CochlScene-SceneClassification/test"
# ALL_TASK="${ALL_TASK} NonSpeech7k-EventClassification/test"

# # # ===== zero-shot =====
# ALL_TASK="${ALL_TASK} CREMA-D-EmotionClassification/train"
# ALL_TASK="${ALL_TASK} ravdess-EmotionClassification/train"
# ALL_TASK="${ALL_TASK} UrbanSound8K-EventClassification/train"
# ALL_TASK="${ALL_TASK} GTZAN-GenreClassification/train"
# ALL_TASK="${ALL_TASK} Medley-solos-DB-InstrClassification/test"

for task in ${ALL_TASK}
do 
    OUTFOLDER=${task//\//-}  # replace / into -
    mkdir -p ../outputs/$OUTFOLDER
done

temp=0.0
numbeams=1
ckpt=199

for EXP in $TO_SUBMIT_JOBS
do
    L=${#EXP}
    NAME=$(echo ${EXP} | cut -c 1-$(($L-5)))  # remove last .yaml

    for task in ${ALL_TASK}
    do
        echo "task: $task, config: $NAME, ckpt: $ckpt"

        OUTFOLDER=${task//\//-}
        OUTFILE="../outputs/$OUTFOLDER/$NAME-ckpt${ckpt}.log"
        CKPT_DIR="/lustre/fsw/portfolios/adlr/users/sreyang/flamingo_v2/af2_exp_qwen3b_rotary_all_layers-7b-fixed-sft/run_demo_pretraining_bf16_xattnevery1_msclapcap_win7_ovlp5.25_single16win-4node-qwen3b-rotary-3b-fixed-sft-3/$NAME"
        python -u inference.py \
            -c ../configs/$EXP \
            -t $task \
            -temp $temp \
            -nb $numbeams \
            --ckpt ${ckpt}

    done
    wait
done