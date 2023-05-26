for i in {138..158}
do
    CUDA_VISIBLE_DEVICES=5 python run.py --data Weather --features M --seq_len 201 --pred_len 168 --latent_size 128 --batch_size 32 --patience 10 --Encoder_Muti_Scale --Mean_Var_Model mlp --seed $i
done