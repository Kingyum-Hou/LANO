nohup python -u src/main.py \
    datamodule=ns_v-5 \
    datamodule.task=task3 \
    datamodule.missing_rate=0.5 \
    model=TransolverPro \
    trainer.devices=[0] \
    trainer.max_epochs=500 \
    datamodule.b_train_test=[4,4] \
    tag=enhance_2 \
    >> logs/NSv-5_task3_Transolver_2.log 2>&1 & sleep 5s
