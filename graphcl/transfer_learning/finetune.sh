#### GIN fine-tuning
split=scaffold

CUDA_VISIBLE_DEVICES=0
for model in 60 80 100 
do
for lr in 0.01 0.001 0.0001
do
for dataset in bace bbbp clintox hiv muv sider tox21 toxcast
do
for runseed in 0 1 2 3 4 5 6 7 8 9
do
python finetune.py --input_model_file models_srgcl/srgcl_${model}.pth --lr $lr --split $split --runseed $runseed --gnn_type gin --dataset $dataset --epochs 100
done
done
done
done


