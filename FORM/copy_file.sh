# you need to assign your autokeras path and this dir here
form_dir="./AutoKeras"
your_lib="./site-packages/autokeras/"


file_list=('/engine/tuner.py' '/tuners/greedy.py' '/engine/compute_gradient.py')

for file in ${file_list[@]};
do
    cp $form_dir$file $your_lib$file
done