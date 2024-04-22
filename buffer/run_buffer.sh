CUDA_VISIBLE_DEVICES=0 python buffer_FTD.py --dataset=CIFAR100 --model=ConvNet --train_epochs=50 --num_experts=100 --zca --buffer_path=./buffer --data_path=/data/ --rho_max=0.01 --rho_min=0.01
