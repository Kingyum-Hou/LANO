nohup python -u src/main.py \
    datamodule=ns_v-5 \
    datamodule.task=task3 \
    datamodule.missing_rate=0.5 \
    model=TransolverPro \
    trainer.devices=[2] \
    trainer.max_epochs=500 \
    datamodule.b_train_test=[2,2] \
    tag=Ours \
    >> logs/NSv-5_task3_Transolver_mr=50.log 2>&1 & sleep 5s

nohup python -u src/main.py \
    datamodule=ns_v-5 \
    datamodule.task=task3 \
    datamodule.missing_rate=0.25 \
    model=TransolverPro \
    trainer.devices=[1] \
    trainer.max_epochs=500 \
    datamodule.b_train_test=[2,2] \
    tag=Ours \
    >> logs/NSv-5_task3_Transolver_mr=25.log 2>&1 & sleep 5s &

nohup python -u src/main.py \
    datamodule=ns_v-5 \
    datamodule.task=task3 \
    datamodule.missing_rate=0.05 \
    model=TransolverPro \
    trainer.devices=[0] \
    trainer.max_epochs=500 \
    datamodule.b_train_test=[2,2] \
    tag=Ours \
    >> logs/NSv-5_task3_Transolver_mr=5.log 2>&1 & sleep 5s
