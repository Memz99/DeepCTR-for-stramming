root="/home/hemingzhi/.jupyter/ctr"
table="xtr_v1"
date="20210608"

frag_size=200000
data_file="${root}/data/${table}/${date}"
data_config="${root}/run/preprocess/configs/xtr_base_no_emb.json"
vocab_save_path="${root}/data/vocab/${table}_${date}.pkl"

python preprocess.py \
  --data_file $data_file \
  --data_config $data_config \
  --vocab_save_path $vocab_save_path \
  --frag_size $frag_size
