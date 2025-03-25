# CHECK_EVERY=900
# DURATION_DAYS=10
# CHECK_TOTAL=$((DURATION_DAYS*86400/CHECK_EVERY))
# NEPOCH_PRE=99
# NEPOCH_SFT=159
# NAME="audio-gen-train_audiogen"

# for (( i = 1; i <= $CHECK_TOTAL; i++ )) 
# do
#     RUNNING_JOBS=$(sacct -o JobName%-150,JobID,Partition%-15,State | grep -v inference | grep RUNNING | grep polar | sort)
#     PENDING_JOBS=$(sacct -o JobName%-150,JobID,Partition%-15,State | grep -v inference | grep PENDING | grep polar | sort)

#     for STATE in "RUNNING" "PENDING" "NOT-RUN"
#     do
#         echo "===========${STATE}=========="

#         if [[ ${STATE} == "RUNNING" && ${RUNNING_JOBS} =~ "${NAME}" ]]; then
#             echo ${NAME}
#         elif [[ ${STATE} == "PENDING" && ${PENDING_JOBS} =~ "${NAME}" ]]; then
#             echo ${NAME}
#         elif [[ ${STATE} == "NOT-RUN" && ! ${RUNNING_JOBS} =~ "${NAME}" && ! ${PENDING_JOBS} =~ "${NAME}" ]]; then

#             base_path="/lustre/fsw/portfolios/adlr/users/sreyang/ckpts/stable_llm/harmonai_train/"
#             # Find the last subfolder
#             last_subfolder=$(ls -d "$base_path"*/ | sort -V | tail -n 1)
#             # Find the last checkpoint in the subfolder
#             last_ckpt=$(ls "$last_subfolder/checkpoints/"*.ckpt | sort -V | tail -n 1)
#             echo $last_ckpt
#             sh submit_job.sh "True" $last_ckpt
#             sleep 1
#         fi
#     done
#     echo "============================"
#     sleep $CHECK_EVERY
# done

CHECK_EVERY=900
DURATION_DAYS=10
CHECK_TOTAL=$((DURATION_DAYS*86400/CHECK_EVERY))
NEPOCH_PRE=99
NEPOCH_SFT=159
NAME="eval"

for (( i = 1; i <= $CHECK_TOTAL; i++ )) 
do
    RUNNING_JOBS=$(sacct -o JobName%-150,JobID,Partition%-15,State | grep -v inference | grep RUNNING | grep polar | sort)
    PENDING_JOBS=$(sacct -o JobName%-150,JobID,Partition%-15,State | grep -v inference | grep PENDING | grep polar | sort)

    for STATE in "RUNNING" "PENDING" "NOT-RUN"
    do
        echo "===========${STATE}=========="

        if [[ ${STATE} == "RUNNING" && ${RUNNING_JOBS} =~ "${NAME}" ]]; then
            echo ${NAME}
        elif [[ ${STATE} == "PENDING" && ${PENDING_JOBS} =~ "${NAME}" ]]; then
            echo ${NAME}
        elif [[ ${STATE} == "NOT-RUN" && ! ${RUNNING_JOBS} =~ "${NAME}" && ! ${PENDING_JOBS} =~ "${NAME}" ]]; then
            sh submit.sh
            sleep 1
        fi
    done
    echo "============================"
    sleep $CHECK_EVERY
done