# for DS in COLLAB
for DS in COLLAB MUTAG PROTEINS REDDIT-BINARY DD IMDB-BINARY NCI1 REDDIT-MULTI-5K # 30: tmux_3
do
    python gcl_gcl_method6_temperature.py --aug hybrid --epochs 500 --log_interval 10 --DS $DS --seed 0 --exp_factor 0.7
done