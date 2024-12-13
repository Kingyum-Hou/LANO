# test is IPOT run successfully
nohup python -u src/main.py \
    datamodule.task=task0 \
    datamodule.missing_rate=0. \
    model=IPOT \
    trainer.devices=[1] \
    >> logs/test_origin-task.log 2>&1 & sleep 10s

nohup python -u src/main.py \
    datamodule.task=task1 \
    datamodule.missing_rate=0.1 \
    model=IPOT \
    trainer.devices=[2] \
    >> logs/test_task1.log 2>&1 & sleep 10s
