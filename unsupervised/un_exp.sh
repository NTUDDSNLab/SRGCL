# datasets0=('MUTAG' 'PROTEINS' 'REDDIT-BINARY' 'DD' 'IMDB-BINARY' 'NCI1' 'COLLAB' 'REDDIT-MULTI-5K')
datasets0=('MUTAG')
for dataset in ${datasets0[*]}
do
    bash un_exp_rep.sh ${dataset} 0
done