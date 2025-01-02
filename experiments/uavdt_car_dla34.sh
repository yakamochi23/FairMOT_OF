cd src
python train.py mot --exp_id uavdt_car_dla34 --load_model '../models/ctdet_coco_dla_2x.pth' --data_cfg '../src/lib/cfg/uavdt_car.json'
cd ..