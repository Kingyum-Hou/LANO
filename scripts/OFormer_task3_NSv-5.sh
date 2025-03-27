nohup python -u src/main.py \
    datamodule=ns_v-5 \
    datamodule.task=task3 \
    datamodule.missing_rate=0.5 \
    model=OFormer \
    trainer.devices=[1] \
    trainer.max_epochs=500 \
    datamodule.b_train_test=[25,25] \
    tag=baseline \
    >> logs/NSv-5_task3_OFORMER_mr=50.log 2>&1 & sleep 5s &

nohup python -u src/main.py \
    datamodule=ns_v-5 \
    datamodule.task=task3 \
    datamodule.missing_rate=0.25 \
    model=OFormer \
    trainer.devices=[2] \
    trainer.max_epochs=500 \
    datamodule.b_train_test=[20,20] \
    tag=baseline \
    >> logs/NSv-5_task3_OFORMER_mr=25.log 2>&1 & sleep 5s &

nohup python -u src/main.py \
    datamodule=ns_v-5 \
    datamodule.task=task3 \
    datamodule.missing_rate=0.05 \
    model=OFormer \
    trainer.devices=[3] \
    trainer.max_epochs=500 \
    datamodule.b_train_test=[10,10] \
    tag=baseline \
    >> logs/NSv-5_task3_OFORMER_mr=5.log 2>&1 & sleep 5s
