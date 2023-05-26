for i in {264..284}
do
    CUDA_VISIBLE_DEVICES=6 python run.py --data ETTh1 --features S --seq_len 201 --pred_len 168 --latent_size 128 --batch_size 32 --patience 10 --Encoder_Muti_Scale --Mean_Var_Model mlp --seed $i
done