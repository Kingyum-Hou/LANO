nohup python -u src/main.py \
    datamodule.task=task1 \
    datamodule.missing_rate=0.05 \
    model=OURS \
    trainer.devices=[4] \
    trainer.max_epochs=1000 \
    datamodule.b_train_test=[6,6] \
    >> logs/ns_task1_OURS_1.log 2>&1 & sleep 10s

nohup python -u src/main.py \
    datamodule.task=task1 \
    datamodule.missing_rate=0.25 \
    model=OURS \
    trainer.devices=[5] \
    trainer.max_epochs=1000 \
    datamodule.b_train_test=[8,8] \
    >> logs/ns_task1_OURS_2.log 2>&1 & sleep 5s

nohup python -u src/main.py \
    datamodule.task=task1 \
    datamodule.missing_rate=0.50 \
    model=OURS \
    trainer.devices=[1] \
    trainer.max_epochs=1000 \
    datamodule.b_train_test=[4,4] \
    >> logs/ns_task1_OURS_3.log 2>&1 & sleep 5s
