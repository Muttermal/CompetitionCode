echo "train model deberta-base"
python trainer.py --config=configs/deberta_base.yaml

echo "train model deberta-v3-large"
python trainer.py --config=configs/deberta_v3_large.yaml

echo "train model longformer-base"
python trainer.py --config=configs/longformer_base.yaml

echo "Done"
