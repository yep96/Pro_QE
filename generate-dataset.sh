path=$1

python create_queries_inductive.py --index_only --reindex --emerge_ratio $2 --dataset=$path

for variable in 0 1 2 3 4 5 6 7 8 9 10 11 12 13; do
  python create_queries_inductive.py --gen_train --save_name --gen_id=$variable --gen_train_num=$3 --dataset=$path
  for type in ee es se; do
    python create_queries_inductive.py --gen_valid --save_name --gen_id=$variable --gen_valid_num=$4 --induc_type=$type --dataset=$path
    python create_queries_inductive.py --gen_test --save_name --gen_id=$variable --gen_test_num=$5 --induc_type=$type --dataset=$path
  done
done
python fusedata.py --dataset=$path

for file in stats.txt entities_emerge.txt entities_train.txt triplets_indexified.txt; do
  cp data/$path/$file data/$path-ind/$file
done