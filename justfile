alias setup := install
alias sync := install
alias update := install

# List all commands by default when typing only `just``
@_default:
    just --list

#########
# CHECK #
#########

# Lint and format
[group('check')]
lint:
    uv run ruff check --fix
    uv run ruff format

# Check types
[group('check')]
typing:
    uv run ty check src

# Run all tests
[group('check')]
test *args:
    uv run pytest {{ args }}

# Check module dependencies
[group('check')]
check-imports:
    uv run tach check

# Check docstrings on tests
[group('check')]
check-testdocs:
    uv run interrogate tests

# Check cognitive complexity
[group('check')]
check-complexity:
    uv run complexipy

# Run linting, formatting, tests and type-checking
[group('check')]
check-all: lint test typing check-imports check-testdocs check-complexity

##############
# LIFE CYCLE #
##############

# Bring environment up-to-date
[group('lifecycle')]
install:
    @-[ -d .git ] || git init
    @-cp -n .env.example .env
    @-mkdir -p logs
    uv sync
    uv run prek install --install-hooks --overwrite --no-progress

# Reset environment and all cache files
[group('lifecycle')]
clean:
    uvx pyclean . -d all
    uvx prek uninstall --no-progress
    -rm -rf .venv

# Setup environment from scratch
[group('lifecycle')]
fresh: clean install

# Upgrade python and all dependencies
[group('lifecycle')]
upgrade python='3.14': (_upgrade_python python)
    uv sync --upgrade
    uv run prek auto-update --no-progress

[arg('python', pattern='^3\.[1-9]\d+$')]
@_upgrade_python python:
    -brew upgrade --quiet uv 2> /dev/null || uv self update --quiet
    -uv python upgrade {{ python }}
    uv python find {{ python }} --show-version --no-project 1> /dev/null
    perl -i -pe 's/(^requires-python\s*=\s*)"[^"]*"/\1"=={{ python }}.*"/' pyproject.toml
    perl -i -pe "s/(^upgrade\s*python\s*)=\s*'3\.[1-9]\d+'/\1='{{ python }}'/" "{{ justfile() }}"
    uv python pin $(uv python find {{ python }} --show-version --no-project)

#######
# RUN #
#######

# Run default model training
[group('run')]
train *args:
    uv run train {{ args }}

# Run default model inference
[group('run')]
predict *args:
    uv run predict {{ args }}

# Run an experiment
[group('run')]
experiment name:
    uv run evaluate +experiment={{ name }}

##########
# CREATE #
##########

# Create a new analysis notebook
[group('create')]
add-notebook name:
    cp -i notebooks/template.ipynb notebooks/{{ name }}.ipynb
    code notebooks/{{ name }}.ipynb

# Create an experiment config
[group('create')]
add-experiment name:
    printf "# @package _global_\n\nexp_name: {{ name }}\n" > config/experiment/{{ name }}.yaml
    code -g config/experiment/{{ name }}.yaml:3:11
