set -x
root="/home/hemingzhi/.jupyter/ctr"
table="xtr_v2"
date="20210609"

is_encode=0
is_config=0
is_split=1

data_file="${root}/data/${table}/${date}"
feat_info_json="${root}/run/preprocess/configs/${table}.json"
vocab_save_path="${root}/data/vocab/${table}/${date}"
config_save_path="${root}/data/configs/${table}"

# encode
vocab_dataset_size=0.85
topk=500000

# split
frag_size=200000
train_dataset_size=0.9
eval_dataset_size=0.1

python iterable_preprocess.py \
  --is_encode $is_encode \
  --is_config $is_config \
  --is_split $is_split \
  --data_file $data_file \
  --feat_info_json $feat_info_json \
  --vocab_save_path $vocab_save_path \
  --config_save_path $config_save_path \
  --vocab_dataset_size $vocab_dataset_size \
  --topk $topk \
  --frag_size $frag_size \
  --train_dataset_size $train_dataset_size \
  --eval_dataset_size $eval_dataset_size
