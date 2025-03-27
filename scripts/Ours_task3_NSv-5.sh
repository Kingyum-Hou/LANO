nohup python -u src/main.py \
    datamodule=ns_v-5 \
    datamodule.task=task3 \
    datamodule.missing_rate=0.5 \
    model=Ours \
    trainer.devices=[0] \
    trainer.max_epochs=500 \
    datamodule.b_train_test=[2,2] \
    callback.model_checkpoint.monitor=train/full_loss \
    callback.early_stopping.monitor=train/full_loss \
    tag=Ours_noreg \
    >> logs/NSv-5_task3_Ours_mr=50_noreg.log 2>&1 & sleep 5s &

nohup python -u src/main.py \
    datamodule=ns_v-5 \
    datamodule.task=task3 \
    datamodule.missing_rate=0.25 \
    model=Ours \
    trainer.devices=[1] \
    trainer.max_epochs=500 \
    datamodule.b_train_test=[2,2] \
    callback.model_checkpoint.monitor=train/full_loss \
    callback.early_stopping.monitor=train/full_loss \
    tag=Ours_noreg \
    >> logs/NSv-5_task3_Ours_mr=25_noreg.log 2>&1 & sleep 5s &

nohup python -u src/main.py \
    datamodule=ns_v-5 \
    datamodule.task=task3 \
    datamodule.missing_rate=0.05 \
    model=Ours \
    trainer.devices=[2] \
    trainer.max_epochs=500 \
    datamodule.b_train_test=[2,2] \
    callback.model_checkpoint.monitor=train/full_loss \
    callback.early_stopping.monitor=train/full_loss \
    tag=Ours_noreg \
    >> logs/NSv-5_task3_Ours_mr=5_noreg.log 2>&1 & sleep 5s &
