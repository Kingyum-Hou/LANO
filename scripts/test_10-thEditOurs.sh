nohup python -u src/main.py \
    datamodule=ns_v-5 \
    datamodule.task=task3 \
    datamodule.missing_rate=0.5 \
    model=Ours \
    trainer.devices=[0] \
    trainer.max_epochs=500 \
    datamodule.n_train_test=[500,100] \
    datamodule.b_train_test=[4,4] \
    callback.model_checkpoint.monitor=train/full_loss \
    callback.early_stopping.monitor=train/full_loss \
    tag=test10 \
    >> logs/NSv-5_task3_Ours_mr=50_10.log 2>&1 & sleep 5s
