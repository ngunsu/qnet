# Country
th nirscenes_triplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq country -net ../nets/train_only/pnnet_256_raw.t7 -false_pair rgb -save results/1 -max_iter 100 -learning_rate 1.4
th nirscenes_triplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq country -net ../nets/train_only/pnnet_256_raw.t7 -false_pair nir -save results/2 -max_iter 100 -learning_rate 1.4
th nirscenes_triplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq country -net ../nets/train_only/pnnet_256_raw.t7 -false_pair random -save results/3 -max_iter 100 -learning_rate 1.4

# Field
th nirscenes_triplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq field -net ../nets/train_only/pnnet_256_raw.t7 -false_pair rgb -save results/4 -max_iter 100 -learning_rate 1.4
th nirscenes_triplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq field -net ../nets/train_only/pnnet_256_raw.t7 -false_pair nir -save results/5 -max_iter 100 -learning_rate 1.4
th nirscenes_triplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq field -net ../nets/train_only/pnnet_256_raw.t7 -false_pair random -save results/6 -max_iter 100 -learning_rate 1.4

# Forest
th nirscenes_triplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq forest -net ../nets/train_only/pnnet_256_raw.t7 -false_pair rgb -save results/7 -max_iter 100 -learning_rate 1.4
th nirscenes_triplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq forest -net ../nets/train_only/pnnet_256_raw.t7 -false_pair nir -save results/8 -max_iter 100 -learning_rate 1.4
th nirscenes_triplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq forest -net ../nets/train_only/pnnet_256_raw.t7 -false_pair random -save results/9 -max_iter 100 -learning_rate 1.4

# Mountain
th nirscenes_triplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq mountain -net ../nets/train_only/pnnet_256_raw.t7 -false_pair rgb -save results/10 -max_iter 100 -learning_rate 1.4
th nirscenes_triplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq mountain -net ../nets/train_only/pnnet_256_raw.t7 -false_pair nir -save results/11 -max_iter 100 -learning_rate 1.4
th nirscenes_triplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq mountain -net ../nets/train_only/pnnet_256_raw.t7 -false_pair random -save results/12 -max_iter 100 -learning_rate 1.4

# Oldbuilding
th nirscenes_triplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq oldbuilding -net ../nets/train_only/pnnet_256_raw.t7 -false_pair rgb -save results/13 -max_iter 100 -learning_rate 1.4
th nirscenes_triplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq oldbuilding -net ../nets/train_only/pnnet_256_raw.t7 -false_pair nir -save results/14 -max_iter 100 -learning_rate 1.4
th nirscenes_triplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq oldbuilding -net ../nets/train_only/pnnet_256_raw.t7 -false_pair random -save results/15 -max_iter 100 -learning_rate 1.4

# Indoor
th nirscenes_triplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq indoor -net ../nets/train_only/pnnet_256_raw.t7 -false_pair rgb -save results/16 -max_iter 100 -learning_rate 1.4
th nirscenes_triplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq indoor -net ../nets/train_only/pnnet_256_raw.t7 -false_pair nir -save results/17 -max_iter 100 -learning_rate 1.4
th nirscenes_triplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq indoor -net ../nets/train_only/pnnet_256_raw.t7 -false_pair random -save results/18 -max_iter 100 -learning_rate 1.4

# Street
th nirscenes_triplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq street -net ../nets/train_only/pnnet_256_raw.t7 -false_pair rgb -save results/19 -max_iter 100 -learning_rate 1.4
th nirscenes_triplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq street -net ../nets/train_only/pnnet_256_raw.t7 -false_pair nir -save results/20 -max_iter 100 -learning_rate 1.4
th nirscenes_triplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq street -net ../nets/train_only/pnnet_256_raw.t7 -false_pair random -save results/21 -max_iter 100 -learning_rate 1.4

# Urban
th nirscenes_triplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq urban -net ../nets/train_only/pnnet_256_raw.t7 -false_pair rgb -save results/22 -max_iter 100 -learning_rate 1.4
th nirscenes_triplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq urban -net ../nets/train_only/pnnet_256_raw.t7 -false_pair nir -save results/23 -max_iter 100 -learning_rate 1.4
th nirscenes_triplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq urban -net ../nets/train_only/pnnet_256_raw.t7 -false_pair random -save results/24 -max_iter 100 -learning_rate 1.4

# Water
th nirscenes_triplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq water -net ../nets/train_only/pnnet_256_raw.t7 -false_pair rgb -save results/25 -max_iter 100 -learning_rate 1.4
th nirscenes_triplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq water -net ../nets/train_only/pnnet_256_raw.t7 -false_pair nir -save results/26 -max_iter 100 -learning_rate 1.4
th nirscenes_triplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq water -net ../nets/train_only/pnnet_256_raw.t7 -false_pair random -save results/27 -max_iter 100 -learning_rate 1.4

