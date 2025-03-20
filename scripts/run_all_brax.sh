for value in {1..10}
do 
    echo $value

    # Kheperax
    python -m main main ga env=kheperax seed=$RANDOM
    python -m main main jedi env=kheperax seed=$RANDOM
    python -m main main me env=kheperax seed=$RANDOM
    python -m main main me env=kheperax env.task.bd_extractor="bumper_contacts" seed=$RANDOM
    python -m main main me env=kheperax env.task.bd_extractor="laser_measures" env.task.grid_shape=[10,10,10] env.task.num_descriptors=3 env.task.max_bd=0.2 seed=$RANDOM
    python -m main main me env=kheperax env.task.bd_extractor="random" seed=$RANDOM
    python -m main main me env=kheperax extinction=true seed=$RANDOM
    python -m main main aurora env=kheperax loss_type="mse" seed=$RANDOM
    python -m main main aurora env=kheperax loss_type="triplet" seed=$RANDOM
    python -m main main aurora env=kheperax loss_type="mse" extinction=true seed=$RANDOM
    python -m main main aurora env=kheperax loss_type="triplet" extinction=true seed=$RANDOM

    # Walker2D
    python -m main main ga env=brax/walker seed=$RANDOM
    python -m main main td3 env=brax/walker seed=$RANDOM
    python -m main main jedi env=brax/walker seed=$RANDOM
    python -m main main pga_me env=brax/walker seed=$RANDOM
    python -m main main pga_me env=brax/walker extinction=true seed=$RANDOM
    python -m main main pga_me env=brax/walker env.task.bd_extractor="random" seed=$RANDOM 
    python -m main main pga_aurora env=brax/walker loss_type="mse" seed=$RANDOM
    python -m main main pga_aurora env=brax/walker loss_type="triplet" seed=$RANDOM
    python -m main main pga_aurora env=brax/walker loss_type="mse" extinction=true seed=$RANDOM
    python -m main main pga_aurora env=brax/walker loss_type="triplet" extinction=true seed=$RANDOM

    # HalfCheetah
    python -m main main ga env=brax/half_cheetah seed=$RANDOM
    python -m main main td3 env=brax/half_cheetah seed=$RANDOM
    python -m main main jedi env=brax/half_cheetah seed=$RANDOM
    python -m main main pga_me env=brax/half_cheetah seed=$RANDOM
    python -m main main pga_me env=brax/half_cheetah extinction=true seed=$RANDOM
    python -m main main pga_me env=brax/half_cheetah env.task.bd_extractor="random" seed=$RANDOM
    python -m main main pga_aurora env=brax/half_cheetah loss_type="mse" seed=$RANDOM
    python -m main main pga_aurora env=brax/half_cheetah loss_type="triplet" seed=$RANDOM
    python -m main main pga_aurora env=brax/half_cheetah loss_type="mse" extinction=true seed=$RANDOM
    python -m main main pga_aurora env=brax/half_cheetah loss_type="triplet" extinction=true seed=$RANDOM
    

    # AntMaze
    python -m main main ga env=brax/ant_maze seed=$RANDOM
    python -m main main td3 env=brax/ant_maze seed=$RANDOM
    python -m main main jedi env=brax/ant_maze seed=$RANDOM
    python -m main main pga_me env=brax/ant_maze seed=$RANDOM
    python -m main main pga_me env=brax/ant_maze extinction=true seed=$RANDOM
    python -m main main pga_me env=brax/ant_maze env.task.bd_extractor="random" seed=$RANDOM 
    python -m main main pga_aurora env=brax/ant_maze loss_type="mse" seed=$RANDOM
    python -m main main pga_aurora env=brax/ant_maze loss_type="triplet" seed=$RANDOM
    python -m main main pga_aurora env=brax/ant_maze loss_type="mse" extinction=true seed=$RANDOM
    python -m main main pga_aurora env=brax/ant_maze loss_type="triplet" extinction=true seed=$RANDOM

done