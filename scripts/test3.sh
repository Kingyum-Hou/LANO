# test is IPOT run successfully
nohup python -u src/main.py \
    datamodule.task=task0 \
    datamodule.missing_rate=0. \
    model=IPOT \
    trainer.devices=[4] \
    >> logs/test_origin-task_IPOT.log 2>&1 & sleep 10s
