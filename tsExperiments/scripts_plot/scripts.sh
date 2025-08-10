bash tactis.sh 3142 4
bash tempflow_mean_std.sh 3142 4
bash transformer_tempflow_mean_std.sh 3142 4
bash timegrad_mean_std.sh 3142 4
bash amcl_std.sh 3142 4

cd ..

python plot_crypt_grid.py
python plot_crypt.py