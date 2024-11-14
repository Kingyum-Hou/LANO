nohup python -u src/main.py \
    datamodule.task=task1 \
    datamodule.missing_rate=0.05 \
    model=OFORMER \
    trainer.devices=[0] \
    trainer.max_epochs=1000 \
    datamodule.b_train_test=[4,4] \
    >> logs/ns_task1_OFormer_0.05.log 2>&1 & sleep 10s

nohup python -u src/main.py \
    datamodule.task=task1 \
    datamodule.missing_rate=0.25 \
    model=OFORMER\
    trainer.devices=[1] \
    trainer.max_epochs=1000 \
    datamodule.b_train_test=[4,4] \
    >> logs/ns_task1_OFormer_0.25.log 2>&1 & sleep 5s

nohup python -u src/main.py \
    datamodule.task=task1 \
    datamodule.missing_rate=0.50 \
    model=OFORMER \
    trainer.devices=[7] \
    trainer.max_epochs=1000 \
    datamodule.b_train_test=[4,4] \
    >> logs/ns_task1_OFormer_0.5.log 2>&1 & sleep 5s
