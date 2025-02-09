IMAGE="/lustre/fsw/portfolios/adlr/users/zkong/docker/audiolm-0.1/image.sqsh"
NAME=eval
PARTITION="polar,polar3,polar4"
MOUNTS="/home/zkong,/lustre/fsw/portfolios/adlr/users/zkong,/lustre/fsw/portfolios/adlr/users/sreyang,/home/sreyang"

LOGDIR=/lustre/fsw/portfolios/adlr/users/sreyang/flamingo_v2/af2_exp_qwen3b_fixed_rotary_all_layers_logs_infer

# "MMAU/test" "MusicCaps-AudioCaptioning/test" "audiocaps-AudioCaptioning/test" "FSD50k-EventClassification/test"
# Predefined list of strings


STRING_LIST=("Music4All/train")
    
    
# "MusicCaps-AudioCaptioning/test_2")
    
# "Clotho-AQA-AQA/test" "MusicCaps-AudioCaptioning/test" "audiocaps-AudioCaptioning/test" "FSD50k-EventClassification/test" "AudioHalQA/test_compa" "MMAU/test" "AIR-Bench/test" "MuschoMusicQA/test" "CREMA-D-EmotionClassification/train" "ravdess-EmotionClassification/train" "UrbanSound8K-EventClassification/train" "ESC50-EventClassification/train" "DCASE17Task4-SceneClassification/test" "GTZAN-GenreClassification/train" "Medley-solos-DB-InstrClassification/test" "Music-AVQA-AQA_All/test" "MU-LLAMA-AQA/test" "AudioEntailmentQA/test" "AudioEntailmentQA/test_audiocaps" "SongDescriber-AudioCaptioning/train")
    
# "Clotho-v2-AudioCaptioning/test")

# "NSynth-Source/test" "NSynth-Instrument/test" "CochlScene-SceneClassification/test")
    
#"Clotho-AQA-AQA/test" "MusicCaps-AudioCaptioning/test" "audiocaps-AudioCaptioning/test" "FSD50k-EventClassification/test" "AudioHalQA/test_compa" "MMAU/test" "AIR-Bench/test" "MuschoMusicQA/test" "CREMA-D-EmotionClassification/train" "ravdess-EmotionClassification/train" "UrbanSound8K-EventClassification/train" "ESC50-EventClassification/train" "DCASE17Task4-SceneClassification/test" "GTZAN-GenreClassification/train" "Medley-solos-DB-InstrClassification/test" "Music-AVQA-AQA_All/test" "MU-LLAMA-AQA/test" "AudioEntailmentQA/test" "AudioEntailmentQA/test_audiocaps" "SongDescriber-AudioCaptioning/train"
    
#"Clotho-AQA-AQA/test" "MusicCaps-AudioCaptioning/test" "audiocaps-AudioCaptioning/test" "FSD50k-EventClassification/test" "AudioHalQA/test_compa" "MMAU/test" "AIR-Bench/test" "MuschoMusicQA/test" "CREMA-D-EmotionClassification/train" "ravdess-EmotionClassification/train" "UrbanSound8K-EventClassification/train" "ESC50-EventClassification/train" "DCASE17Task4-SceneClassification/test" "GTZAN-GenreClassification/train" "Medley-solos-DB-InstrClassification/test" "Music-AVQA-AQA_All/test" "MU-LLAMA-AQA/test" "AudioEntailmentQA/test" "AudioEntailmentQA/test_audiocaps" "SongDescriber-AudioCaptioning/train")
    
# "Clotho-AQA-AQA/test" "MusicCaps-AudioCaptioning/test" "audiocaps-AudioCaptioning/test" "FSD50k-EventClassification/test" "AudioHalQA/test_compa" "MMAU/test" "AIR-Bench/test" "MuschoMusicQA/test" "CREMA-D-EmotionClassification/train" "ravdess-EmotionClassification/train" "UrbanSound8K-EventClassification/train" "ESC50-EventClassification/train" "DCASE17Task4-SceneClassification/test" "GTZAN-GenreClassification/train" "Medley-solos-DB-InstrClassification/test"

# "CREMA-D-EmotionClassification/train" "ravdess-EmotionClassification/train" "UrbanSound8K-EventClassification/train" "ESC50-EventClassification/train" "DCASE17Task4-SceneClassification/test" "GTZAN-GenreClassification/train" "Medley-solos-DB-InstrClassification/test"
#("Clotho-AQA-AQA/test" "AudioHalQA/test_compa" "MMAU/test" "AIR-Bench/test")

for i in "${STRING_LIST[@]}"; do

    OUTFILE=$LOGDIR/output_$i-2.out

    TASK=""
    TASK="${TASK} $i"

    SUBMIT_SUBPROJECT_NAME="llmservice_fm_audio" submit_job \
        --mounts $MOUNTS \
        --name audio-flamingo-$NAME \
        --duration 4 \
        --partition $PARTITION \
        --gpu 2 \
        --nodes 1 \
        --image $IMAGE \
        --email_mode never \
        --outfile $OUTFILE \
        --logdir $LOGDIR \
        --prolog_command "pip install nnAudio; pip install tokenizers==0.20.3; pip install transformers==4.46.3" \
        --command "sh inference.sh $TASK"
    sleep 30
done

