nohup python -u src/main.py \
    datamodule=ns_v-5 \
    datamodule.task=task3 \
    datamodule.missing_rate=0.5 \
    model=OFORMER_FILLGAP \
    trainer.devices=[0] \
    datamodule.b_train_test=[40,40] \
    model.params_model.scale_factor=4. \
    model.params_model.r=8. \
    tag=scale4_ref8 \
    >> logs/NSv-5_task3_OFORMER-FILLGAP_scale=4_ref=8.log 2>&1 & sleep 5s

wait

nohup python -u src/main.py \
    datamodule=ns_v-5 \
    datamodule.task=task3 \
    datamodule.missing_rate=0.5 \
    model=OFORMER_FILLGAP \
    trainer.devices=[0] \
    datamodule.b_train_test=[40,40] \
    model.params_model.scale_factor=8. \
    model.params_model.r=8 \
    tag=scale8_ref8 \
    >> logs/NSv-5_task3_OFORMER-FILLGAP_scale=8_ref=8.log 2>&1 & sleep 5s &

wait

nohup python -u src/main.py \
    datamodule=ns_v-5 \
    datamodule.task=task3 \
    datamodule.missing_rate=0.5 \
    model=OFORMER_FILLGAP \
    trainer.devices=[0] \
    datamodule.b_train_test=[40,40] \
    model.params_model.scale_factor=4. \
    model.params_model.r=16 \
    tag=scale4_ref16 \
    >> logs/NSv-5_task3_OFORMER-FILLGAP_scale=4_ref=16.log 2>&1 & sleep 5s &

wait

nohup python -u src/main.py \
    datamodule=ns_v-5 \
    datamodule.task=task3 \
    datamodule.missing_rate=0.5 \
    model=OFORMER_FILLGAP \
    trainer.devices=[0] \
    datamodule.b_train_test=[40,40] \
    model.params_model.scale_factor=8 \
    model.params_model.r=16 \
    tag=scale8_ref16 \
    >> logs/NSv-5_task3_OFORMER-FILLGAP_scale=8_ref=16.log 2>&1 & sleep 5s
