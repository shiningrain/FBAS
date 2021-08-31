# you need to assign your autokeras path and this dir here
bk_path_1="/data/zxy/DL_autokeras/1Autokeras/test_codes/FBAS/AutoKeras"
new_name_1="/data/zxy/anaconda3/envs/ourak_test/lib/python3.7/site-packages/autokeras"


file_list=('/engine/tuner.py' '/tuners/greedy.py' '/engine/compute_gradient.py')

for file in ${file_list[@]};
do
    # echo $file
    # echo 1
    cp $bk_path_1$file $new_name_1$file
done