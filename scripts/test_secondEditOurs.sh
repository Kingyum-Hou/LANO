nohup python -u src/main.py \
    datamodule=ns_v-5 \
    datamodule.task=task3 \
    datamodule.missing_rate=0.5 \
    model=Ours \
    trainer.devices=[2] \
    trainer.max_epochs=500 \
    datamodule.b_train_test=[2,2] \
    callback.model_checkpoint.monitor=train/full_loss \
    callback.early_stopping.monitor=train/full_loss \
    tag=test2 \
    >> logs/NSv-5_task3_Ours_mr=50_2.log 2>&1 & sleep 5s
