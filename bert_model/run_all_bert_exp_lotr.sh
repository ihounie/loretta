#export HF_HOME='/data/public/yifanyang'
#export WANDB_START_METHOD="thread"

# GLUE datasets supported in this code (change in $TASK)
# MNLI, SST2, COLA, QQP, QNLI, RTE, MRPC, STSB
# PEFT methods supported in this code (change in $MODE)
# loretta_adp, loretta_rep, lora, adapters (series), prompt, ia3, ptune
# change the number of cuda device by set the $DEVICE

# Test with deberta-base model (use SST2 dataset by default, change the input for $TASK for other tasks)
#MODEL=microsoft/deberta-base TASK=SST2 MODE=ft EPOCH=10 BS=8 LR=1e-6 DEVICE=7 bash finetune.sh
#MODEL=microsoft/deberta-base TASK=SST2 MODE=loretta_adp EPOCH=10 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=microsoft/deberta-base TASK=SST2 MODE=loretta_rep EPOCH=10 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=microsoft/deberta-base TASK=SST2 MODE=lora EPOCH=10 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=microsoft/deberta-base TASK=SST2 MODE=adapters EPOCH=10 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=microsoft/deberta-base TASK=SST2 MODE=prompt EPOCH=10 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=microsoft/deberta-base TASK=SST2 MODE=ia3 EPOCH=10 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=microsoft/deberta-base TASK=SST2 MODE=ptune EPOCH=10 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=microsoft/deberta-base TASK=SST2 MODE=bitfit EPOCH=10 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh

# Test with roberta-base model (use SST2 dataset by default, change the input for $TASK for other tasks)

#MODEL=roberta-large TASK=SST2 MODE=ft EPOCH=10 BS=8 LR=1e-6 DEVICE=7 bash finetune.sh
#MODEL=roberta-large TASK=SST2 MODE=loretta_adp EPOCH=10 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=roberta-large TASK=SST2 MODE=loretta_rep EPOCH=10 BS=8 LR=1e-4 DEVICE=6 bash finetune.sh
#MODEL=roberta-large TASK=SST2 MODE=lora EPOCH=10 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=roberta-large TASK=SST2 MODE=adapters EPOCH=10 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=roberta-large TASK=SST2 MODE=prompt EPOCH=10 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=roberta-large TASK=SST2 MODE=ia3 EPOCH=10 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=roberta-large TASK=SST2 MODE=ptune EPOCH=10 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=roberta-large TASK=SST2 MODE=bitfit EPOCH=10 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
for dataset in 'COLA'
do
    for tensor_rank in 16 4 64
    do
        for learning_rate in 5e-4 1e-3 5e-3 1e-2
        do
            for batch_size in 16 32
            do 
                for lora_alpha in 0.5 1.0 2.0 8.0
                do
                    MODEL=roberta-base TASK=$dataset MODE=lorta EPOCH=20 BS=$batch_size LR=$learning_rate DEVICE=1 RANK=$tensor_rank ALPHA=$lora_alpha bash finetune_lotr.sh
                done
            done
        done
    done
done
