# Analysis Notebooks

Notebooks contain analysis, exploratory, or experimental code.
They may stay self-contained or are refactored to become part of the production code.

They reside under the `notebooks` directory.

## Format

Notebooks follow the general theme of outlying the purpose or goal at the top, walking the reader through individual steps using markdown, and concluding with a result or decision.

Ideally the notebook stays concise to paint a clear picture of the analysis and is committed with cleared outputs to prevent bloating the git repository.

## Create Notebooks

New notebooks are conveniently created from a template using the command palette "Just: Run Recipe" -> "add-notebook" or from the console.

```sh
just add-notebook <name>
```
