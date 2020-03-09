echo 'isort'
isort awesome_glue/*.py
isort allennlpx/**/*.py

echo 'autoflake'
autoflake --in-place --remove-unused-variables -r awesome_glue/**/*.py
autoflake --in-place --remove-unused-variables -r allennlpx/**/*.py
