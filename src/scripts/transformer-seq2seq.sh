# MHQG-PQ
onmt_preprocess -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo -dynamic_dict
onmt_train -data ../../data/mhqg-pq/transformer -save_model ../out/mhqg-pq/transformer-seq2seq   -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8   -encoder_type transformer -decoder_type transformer -position_encoding -train_steps 2000000  -max_generator_batches 2 -dropout 0.1   -batch_size 60 -batch_type tokens -normalization tokens  -accum_count 2   -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2    -max_grad_norm 0 -param_init 0  -param_init_glorot -label_smoothing 0.1 -valid_steps 5000 -save_checkpoint_steps 5000 -gpu_ranks 0 --copy_attn
onmt_translate -model ../out/mhqg-pq/transformer_copy_step_340000.pt -src ../../data/mhqg-pq/src-test.txt -output ../out/mhqg-pq/pred_transformer_copy_step_340000.txt -replace_unk --gpu 0


# MHQG-WQ
onmt_train -data ../../data/mhqg-wq/transformer -save_model ../out/mhqg-wq/transformer-seq2seq   -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8   -encoder_type transformer -decoder_type transformer -position_encoding -train_steps 2000000  -max_generator_batches 2 -dropout 0.1   -batch_size 60 -batch_type tokens -normalization tokens  -accum_count 2   -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2    -max_grad_norm 0 -param_init 0  -param_init_glorot -label_smoothing 0.1 -valid_steps 5000 -save_checkpoint_steps 5000 -gpu_ranks 0 --copy_attn
onmt_translate -model ../out/mhqg-wq/transformer_copy_step_340000.pt -src ../../data/mhqg-wq/src-test.txt -output ../out/mhqg-wq/pred_transformer_copy_step_340000.txt -replace_unk  --gpu 0
