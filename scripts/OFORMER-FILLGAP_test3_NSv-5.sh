#nohup python -u src/main.py \
#    datamodule=ns_v-5 \
#    datamodule.task=task3 \
#    datamodule.missing_rate=0.5 \
#    model=OFORMER_FILLGAP \
#    trainer.devices=[0] \
#    datamodule.b_train_test=[40,40] \
#    model.params_model.sigma=1. \
#    tag=sigma1. \
#    >> logs/NSv-5_task3_OFORMER-FILLGAP_sigma=1.log 2>&1 & sleep 5s

#wait

nohup python -u src/main.py \
    datamodule=ns_v-5 \
    datamodule.task=task3 \
    datamodule.missing_rate=0.5 \
    model=OFORMER_FILLGAP \
    trainer.devices=[0] \
    datamodule.b_train_test=[40,40] \
    model.params_model.sigma=0.5 \
    tag=sigma0.5 \
    >> logs/NSv-5_task3_OFORMER-FILLGAP_sigma=0.5.log 2>&1 & sleep 5s &

wait

nohup python -u src/main.py \
    datamodule=ns_v-5 \
    datamodule.task=task3 \
    datamodule.missing_rate=0.5 \
    model=OFORMER_FILLGAP \
    trainer.devices=[0] \
    datamodule.b_train_test=[40,40] \
    model.params_model.sigma=0.05 \
    tag=sigma0.05 \
    >> logs/NSv-5_task3_OFORMER-FILLGAP_sigma=0.05.log 2>&1 & sleep 5s &

wait

nohup python -u src/main.py \
    datamodule=ns_v-5 \
    datamodule.task=task3 \
    datamodule.missing_rate=0.5 \
    model=OFORMER_FILLGAP \
    trainer.devices=[0] \
    datamodule.b_train_test=[40,40] \
    model.params_model.sigma=2 \
    tag=sigma2 \
    >> logs/NSv-5_task3_OFORMER-FILLGAP_sigma=2.log 2>&1 & sleep 5s
