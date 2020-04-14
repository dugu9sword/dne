echo 'Running autoflake...'
autoflake --in-place --remove-all-unused-imports --ignore-init-module-imports --recursive awesome_glue/
autoflake --in-place --remove-all-unused-imports --ignore-init-module-imports --recursive allennlpx/

# echo 'isort'
# isort awesome_glue/*.py
# isort allennlpx/**/*.py
