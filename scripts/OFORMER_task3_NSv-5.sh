nohup python -u src/main.py \
    datamodule=ns_v-5 \
    datamodule.task=task3 \
    datamodule.missing_rate=0.5 \
    model=OFORMER\
    trainer.devices=[0] \
    trainer.max_epochs=500 \
    datamodule.b_train_test=[50,50] \
    tag=baseline \
    >> logs/NSv-5_task3_OFORMER_baseline.log 2>&1 & sleep 5s
