export PYTHONPATH=$PYTHONPATH:$(pwd)

# MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3 --diffusion_steps 4000 --noise_schedule cosine "

CUDA_VISIBLE_DEVICES=3 mpiexec -n 1 python image_adapt/scripts/image_sample.py $MODEL_FLAGS \
                            --batch_size 256 --num_samples 10000 --timestep_respacing 400 \
                            --model_path ckpt/cifar10_uncond_50M_500K.pt --base_samples dataset/cifar10c \
                            --D 4 --N 200 --scale 6\
                            --corruption elastic_transform --severity 3 \
                            --save_dir /data/mrigankr/vlr-project/cifar10c/generated/

CUDA_VISIBLE_DEVICES=3 mpiexec -n 1 python image_adapt/scripts/image_sample.py $MODEL_FLAGS \
                            --batch_size 256 --num_samples 10000 --timestep_respacing 400 \
                            --model_path ckpt/cifar10_uncond_50M_500K.pt --base_samples dataset/cifar10c \
                            --D 4 --N 200 --scale 6\
                            --corruption fog --severity 3 \
                            --save_dir /data/mrigankr/vlr-project/cifar10c/generated/

CUDA_VISIBLE_DEVICES=3 mpiexec -n 1 python image_adapt/scripts/image_sample.py $MODEL_FLAGS \
                            --batch_size 256 --num_samples 10000 --timestep_respacing 400 \
                            --model_path ckpt/cifar10_uncond_50M_500K.pt --base_samples dataset/cifar10c \
                            --D 4 --N 200 --scale 6\
                            --corruption frost --severity 3 \
                            --save_dir /data/mrigankr/vlr-project/cifar10c/generated/

CUDA_VISIBLE_DEVICES=3 mpiexec -n 1 python image_adapt/scripts/image_sample.py $MODEL_FLAGS \
                            --batch_size 256 --num_samples 10000 --timestep_respacing 400 \
                            --model_path ckpt/cifar10_uncond_50M_500K.pt --base_samples dataset/cifar10c \
                            --D 4 --N 200 --scale 6\
                            --corruption gaussian_noise --severity 3 \
                            --save_dir /data/mrigankr/vlr-project/cifar10c/generated/

CUDA_VISIBLE_DEVICES=3 mpiexec -n 1 python image_adapt/scripts/image_sample.py $MODEL_FLAGS \
                            --batch_size 256 --num_samples 10000 --timestep_respacing 400 \
                            --model_path ckpt/cifar10_uncond_50M_500K.pt --base_samples dataset/cifar10c \
                            --D 4 --N 200 --scale 6\
                            --corruption glass_blur --severity 3 \
                            --save_dir /data/mrigankr/vlr-project/cifar10c/generated/

CUDA_VISIBLE_DEVICES=3 mpiexec -n 1 python image_adapt/scripts/image_sample.py $MODEL_FLAGS \
                            --batch_size 256 --num_samples 10000 --timestep_respacing 400 \
                            --model_path ckpt/cifar10_uncond_50M_500K.pt --base_samples dataset/cifar10c \
                            --D 4 --N 200 --scale 6\
                            --corruption impulse_noise --severity 3 \
                            --save_dir /data/mrigankr/vlr-project/cifar10c/generated/

CUDA_VISIBLE_DEVICES=3 mpiexec -n 1 python image_adapt/scripts/image_sample.py $MODEL_FLAGS \
                            --batch_size 256 --num_samples 10000 --timestep_respacing 400 \
                            --model_path ckpt/cifar10_uncond_50M_500K.pt --base_samples dataset/cifar10c \
                            --D 4 --N 200 --scale 6\
                            --corruption jpeg_compression --severity 3 \
                            --save_dir /data/mrigankr/vlr-project/cifar10c/generated/

CUDA_VISIBLE_DEVICES=3 mpiexec -n 1 python image_adapt/scripts/image_sample.py $MODEL_FLAGS \
                            --batch_size 256 --num_samples 10000 --timestep_respacing 400 \
                            --model_path ckpt/cifar10_uncond_50M_500K.pt --base_samples dataset/cifar10c \
                            --D 4 --N 200 --scale 6\
                            --corruption motion_blur --severity 3 \
                            --save_dir /data/mrigankr/vlr-project/cifar10c/generated/
