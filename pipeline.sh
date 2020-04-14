rm -rf train.logs/
rm -rf attack.logs/
alchemist --task=train
alchemist --task=attack
