nohup python -u src/main.py \
    datamodule.task=task1 \
    datamodule.missing_rate=0.05 \
    model=MIONET \
    trainer.devices=[4] \
    trainer.max_epochs=1000 \
    datamodule.b_train_test=[32,32] \
    >> logs/ns_task1_MIONet_deeper.log 2>&1 & sleep 10s

nohup python -u src/main.py \
    datamodule.task=task1 \
    datamodule.missing_rate=0.25 \
    model=MIONET \
    trainer.devices=[6] \
    trainer.max_epochs=1000 \
    datamodule.b_train_test=[32,32] \
    >> logs/ns_task1_MIONet_2_deeper.log 2>&1 & sleep 5s

nohup python -u src/main.py \
    datamodule.task=task1 \
    datamodule.missing_rate=0.50 \
    model=MIONET \
    trainer.devices=[7] \
    trainer.max_epochs=1000 \
    datamodule.b_train_test=[32,32] \
    >> logs/ns_task1_MIONet_deeper.log 2>&1 & sleep 5s
