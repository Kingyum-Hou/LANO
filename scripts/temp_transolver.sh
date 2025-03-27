nohup python -u src/main.py \
    datamodule=ns_v-5 \
    datamodule.task=task3 \
    datamodule.missing_rate=0.5 \
    model=TransolverPro \
    trainer.devices=[2] \
    trainer.max_epochs=500 \
    datamodule.b_train_test=[2,2] \
    callback.model_checkpoint.monitor=train/full_loss \
    callback.early_stopping.monitor=train/full_loss \
    tag=Transolver_reg \
    model.params_model.alpha=1e-4 \
    >> logs/NSv-5_task3_Transolver_mr=50_reg.log 2>&1 & sleep 5s