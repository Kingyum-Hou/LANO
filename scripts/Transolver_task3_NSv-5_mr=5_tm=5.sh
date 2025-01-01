nohup python -u src/main.py \
    datamodule=ns_v-5 \
    datamodule.task=task3 \
    datamodule.missing_rate=0.05 \
    model=TransolverPro \
    trainer.devices=[1] \
    trainer.max_epochs=500 \
    datamodule.b_train_test=[2,2] \
    callback.model_checkpoint.monitor=train/full_loss \
    callback.early_stopping.monitor=train/full_loss \
    tag=Ours_tm5 \
    >> logs/NSv-5_task3_Transolver_mr=5_tm=5.log 2>&1 & sleep 5s
