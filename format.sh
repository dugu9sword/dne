echo 'Running autoflake...'
autoflake --in-place --remove-all-unused-imports --recursive awesome_glue/
autoflake --in-place --remove-all-unused-imports --recursive allennlpx/

# echo 'isort'
# isort awesome_glue/*.py
# isort allennlpx/**/*.py