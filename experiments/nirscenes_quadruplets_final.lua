# Country
th nirscenes_quadruplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq country -net ../nets/train_only/quadruplets_4n_1p_256_raw.t7 -save results/1 -max_iter 1000 -learning_rate 1.1

# Field
th nirscenes_quadruplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq field -net ../nets/train_only/quadruplets_4n_1p_256_raw.t7  -save results/2 -max_iter 1000 -learning_rate 1.1

# Forest
th nirscenes_quadruplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq forest -net  ../nets/train_only/quadruplets_4n_1p_256_raw.t7  -save results/3 -max_iter 1000 -learning_rate 1.1

# Indoor
th nirscenes_quadruplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq indoor -net ../nets/train_only/quadruplets_4n_1p_256_raw.t7 -save results/4 -max_iter 1000 -learning_rate 1.1

# Mountain
th nirscenes_quadruplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq mountain -net ../nets/train_only/quadruplets_4n_1p_256_raw.t7  -save results/5 -max_iter 1000 -learning_rate 1.1

# Oldbuilding
th nirscenes_quadruplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq oldbuilding -net  ../nets/train_only/quadruplets_4n_1p_256_raw.t7  -save results/6 -max_iter 1000 -learning_rate 1.1

# Street
th nirscenes_quadruplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq street -net  ../nets/train_only/quadruplets_4n_1p_256_raw.t7  -save results/7 -max_iter 1000 -learning_rate 1.1

# Urban
th nirscenes_quadruplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq urban -net  ../nets/train_only/quadruplets_4n_1p_256_raw.t7 -save results/8 -max_iter 1000 -learning_rate 1.1

# Water
th nirscenes_quadruplets_train.lua -dataset_path /media/cristhian/Storage01/datasets/nirscenes -seq water -net  ../nets/train_only/quadruplets_4n_1p_256_raw.t7  -save results/9 -max_iter 1000 -learning_rate 1.1
