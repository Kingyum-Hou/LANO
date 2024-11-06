# test is IPOT run successfully
nohup python -u src/main.py \
    datamodule.task=task0 \
    datamodule.missing_rate=0. \
    model=FNO \
    trainer.devices=[3] \
    >> logs/test_origin-task_FNO.log 2>&1 & sleep 10s
