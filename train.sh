# SUN360

# SWINR_cos
CUDA_VISIBLE_DEVICES=3 python wire_image_denoise.py \
                        --omega_0 10.0 \
                        --sigma_0 1.0 \
                        --model swinr \
                        --skip \
                        --niters=2000 \
                        --panorama_idx 10 \
                        --normalize \
                        --plot \
                        --lr 0.0003 \
                        --hidden_features 512 \
                        --hidden_layers 2 \
                        --freq_enc_type cos

# SWINR_sin
CUDA_VISIBLE_DEVICES=6 python wire_image_denoise.py \
                        --omega_0 30.0 \
                        --sigma_0 10.0 \
                        --model swinr \
                        --skip \
                        --niters=2000 \
                        --panorama_idx 10 \
                        --normalize \
                        --plot \
                        --lr 0.0003 \
                        --hidden_features 512 \
                        --hidden_layers 2 \
                        --freq_enc_type sin

# Gauss
CUDA_VISIBLE_DEVICES=0 python wire_image_denoise.py \
                        --gauss_scale 4.0 \
                        --model gauss \
                        --skip \
                        --niters=1 \
                        --panorama_idx 10 \
                        --normalize \
                        --plot \
                        --lr 0.0003 \
                        --hidden_features 512 \
                        --hidden_layers 2

# SIREN
CUDA_VISIBLE_DEVICES=0 python wire_image_denoise.py \
                        --omega_0 10.0 \
                        --model siren \
                        --skip \
                        --niters=2000 \
                        --panorama_idx 10 \
                        --normalize \
                        --plot \
                        --lr 0.0003 \
                        --hidden_features 512 \
                        --hidden_layers 6