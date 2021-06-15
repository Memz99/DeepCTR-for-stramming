set -x
root="/home/hemingzhi/.jupyter/ctr"
do="train"

epoch=30
train_batch_size=4800

task="1700w_epoch30_batch_size4800_CE"
table="xtr_base"

train_date="20210608_filtered"
train_path="${root}/data/${table}/${train_date}_train_splits"
info_path="${root}/data/vocab/${table}_${train_date}.pkl"
checkpoint_save_path="${root}/result/${table}/${train_date}/${task}_checkpoint"

eval_date="20210609_keys_in_0608"
eval_path="${root}/data/${table}/${eval_date}_eval_splits"
checkpoint_load_path="${root}/result/xtr_base/${train_date}/${task}_checkpoint"

if [ "$do" = "train" ]
then
  log_path="${root}/result/${table}/${train_date}/${task}_${do}_log"
elif [ "$do" = "eval" ]
then
  log_path="${root}/result/${table}/${eval_date}/${task}_${do}_log"
fi


python ./main_test.py \
  --root $root \
  --do $do \
  --train_path $train_path --eval_path $eval_path \
  --log_path $log_path --info_path $info_path \
  --checkpoint_load_path $checkpoint_load_path  \
  --checkpoint_save_path $checkpoint_save_path \
  --epoch $epoch --train_batch_size $train_batch_size

