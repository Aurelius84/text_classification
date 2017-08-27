# coding:utf-8


class params:

    label_size = 1999
    learning_rate = 0.01
    batch_size = 128
    decay_steps = 20000  # how many steps before decay learning rate
    decay_rate = 0.8  # rate of decay for learning rate
    num_sampled = 50  # number of noise sampling
    ckpt_dir = "fastText_checkpoints/"  # checkpoint location for the model
    sentence_size = 200  # max sentence size
    embed_size = 100
    is_training = True  # is training.true:training, false:testing/inference
    num_epoches = 15  # numbers of training epoch
    validate_every = 10  # validate every validate every epoch
    use_embedding = True  # whether to use embedding or not
    cache_path = "fastText_checkpoints/data_cache.pik"  # checkpoint location for the model
