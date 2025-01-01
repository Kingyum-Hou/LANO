nohup python -u src/main.py \
    datamodule=ns_v-5 \
    datamodule.task=task3 \
    datamodule.missing_rate=0.5 \
    model=OFormer\
    trainer.devices=[3] \
    trainer.max_epochs=500 \
    datamodule.b_train_test=[25,25] \
    callback.model_checkpoint.monitor=train/full_loss \
    callback.early_stopping.monitor=train/full_loss \
    tag=baseline_original \
    >> logs/NSv-5_task3_OFORMER_baseline_mr=50_original_reRun.log 2>&1 & sleep 5s &

nohup python -u src/main.py \
    datamodule=ns_v-5 \
    datamodule.task=task3 \
    datamodule.missing_rate=0.25 \
    model=OFormer\
    trainer.devices=[4] \
    trainer.max_epochs=500 \
    datamodule.b_train_test=[20,20] \
    callback.model_checkpoint.monitor=train/full_loss \
    callback.early_stopping.monitor=train/full_loss \
    tag=baseline_original \
    >> logs/NSv-5_task3_OFORMER_baseline_mr=25_original_reRun.log 2>&1 & sleep 5s &

nohup python -u src/main.py \
    datamodule=ns_v-5 \
    datamodule.task=task3 \
    datamodule.missing_rate=0.05 \
    model=OFormer\
    trainer.devices=[5] \
    trainer.max_epochs=500 \
    datamodule.b_train_test=[10,10] \
    callback.model_checkpoint.monitor=train/full_loss \
    callback.early_stopping.monitor=train/full_loss \
    tag=baseline_original \
    >> logs/NSv-5_task3_OFORMER_baseline_mr=5_original_reRun.log 2>&1 & sleep 5s

#nohup python -u src/main.py \
#    datamodule=ns_v-5 \
#    datamodule.task=task3 \
#    datamodule.missing_rate=0.50 \
#    model=OFormer_small\
#    trainer.devices=[3] \
#    trainer.max_epochs=500 \
#    datamodule.b_train_test=[20,20] \
#    callback.model_checkpoint.monitor=train/full_loss \
#    callback.early_stopping.monitor=train/full_loss \
#    tag=baseline_small \
#    >> logs/NSv-5_task3_OFORMER_baseline_mr=50_small.log 2>&1 & sleep 5s