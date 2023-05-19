CUDA_VISIBLE_DEVICES=0 python run.py --data Electricity  --features S  --seq_len 201  --pred_len 24  --latent_size 128  --batch_size 32  --patience 10 --Encoder_Muti_Scale --Decoder_Muti_Scale --Mean_Var_Model Conv

CUDA_VISIBLE_DEVICES=0 python run.py --data Electricity  --features S  --seq_len 201  --pred_len 48  --latent_size 128  --batch_size 32  --patience 10 --Encoder_Muti_Scale --Decoder_Muti_Scale --Mean_Var_Model Conv

CUDA_VISIBLE_DEVICES=0 python run.py --data Electricity  --features S  --seq_len 201  --pred_len 168  --latent_size 128  --batch_size 32  --patience 10 --Encoder_Muti_Scale --Decoder_Muti_Scale --Mean_Var_Model Conv

CUDA_VISIBLE_DEVICES=0 python run.py --data Electricity  --features S  --seq_len 201  --pred_len 336  --latent_size 128  --batch_size 32  --patience 10 --Encoder_Muti_Scale --Decoder_Muti_Scale --Mean_Var_Model Conv

CUDA_VISIBLE_DEVICES=0 python run.py --data Electricity  --features S  --seq_len 201  --pred_len 720  --latent_size 128  --batch_size 32  --patience 10 --Encoder_Muti_Scale --Decoder_Muti_Scale --Mean_Var_Model Conv

