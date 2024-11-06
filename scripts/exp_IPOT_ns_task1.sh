nohup python -u src/main.py \
    datamodule.task=task1 \
    datamodule.missing_rate=0.05 \
    model=IPOT \
    trainer.devices=[0] \
    scheduler.name=OneCycleLR \
    >> logs/ns_task1_IPOT_1.log 2>&1 & sleep 10s

nohup python -u src/main.py \
    datamodule.task=task1 \
    datamodule.missing_rate=0.25 \
    model=IPOT \
    trainer.devices=[1] \
    scheduler.name=OneCycleLR \
    optim.lr=1e-4 \
    datamodule.b_train_test=[10,10] \
    seed=12340 \
    >> logs/ns_task1_IPOT_2.log 2>&1 & sleep 5s

nohup python -u src/main.py \
    datamodule.task=task1 \
    datamodule.missing_rate=0.50 \
    model=IPOT \
    trainer.devices=[2] \
    scheduler.name=OneCycleLR \
    optim.lr=1e-4 \
    datamodule.b_train_test=[10,10] \
    seed=12340 \
    >> logs/ns_task1_IPOT_3.log 2>&1 & sleep 5s
