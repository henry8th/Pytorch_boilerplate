from data import custom_collate

class hparams:

    dataloader = {
        'batch_size':32,
        'collate_fn':custom_collate,
        'num_workers':4,
        'pin_memory': True,
    }


    training = {
        'lr':0.0001,
    }

    trainer = {
        'gpus':1,
        'check_val_every_n_epoch':10,
    }
        