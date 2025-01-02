cd src
python train.py mot --exp_id visdrone_pedi_ft_mix_dla34 --load_model '../models/fairmot_dla34.pth' --data_cfg '../src/lib/cfg/visdrone_pedi.json'
cd ..