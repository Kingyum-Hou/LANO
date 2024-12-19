nohup python -u src/main.py \
    datamodule=ns_v-5 \
    datamodule.task=task3 \
    datamodule.missing_rate=0.5 \
    model=OURS \
    trainer.devices=[0] \
    trainer.max_epochs=500 \
    datamodule.b_train_test=[5,5] \
    model.params_model.scale_factor=2. \
    model.params_model.r=64 \
    tag=s2r64 \
    >> logs/NSv-5_task3_OURS_s2r64.log 2>&1 & sleep 5s
