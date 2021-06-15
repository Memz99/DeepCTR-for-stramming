set -x
root="/Users/hemingzhi/Documents/Projects/ctr"
is_local=True
do="train"
task="1700w_epoch1"
table="xtr_base"

train_date="20210527_filtered"
train_path="${root}/data/${table}/${train_date}_train_splits"
data_info_path="${root}/data/vocab/${table}_${train_date}.pkl"
train_info_path="${root}/run/estimator/configs/xtr_base_no_sparse.json"

eval_date="20210528_filtered"
eval_path="${root}/data/${table}/${eval_date}_eval_splits"
checkpoint_load_path="${root}/result/xtr_base/${train_date}/${task}_train/checkpoint"

if [ $do == "train" ]
then
  save_path="${root}/result/${table}/${train_date}/${task}_${do}"
elif [ $do == "eval" ]
then
  save_path="${root}/result/${table}/${eval_date}/${task}_${do}"
fi

python ./main_test.py \
  --root $root --is_local $is_local \
  --do $do \
  --train_path $train_path --eval_path $eval_path \
  --data_info_path $data_info_path \
  --train_info_path $train_info_path \
  --checkpoint_load_path $checkpoint_load_path  \
  --save_path $save_path
