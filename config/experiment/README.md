# Experiments

The experiments selectively overwrite the default configuration.

The experiment configs must reside in `config/experiment`.

## Run experiments

There are several ways of running experiments.

1. Run the **[Experiment]** debugger in VS-Code from "Run and Debug" in the side bar.
1. Use just from the command palette in VS-Code: "Run Just Recipe" -> "experiment".
1. From the console with the following command. Note the use of `+`.

    ```sh
    uv run project +experiment=experiment_name
    ```

For all options the experiment is starting in the dev-environment, i.e. `ENV` is "dev".

## Create experiments

Configs for new experiments are conveniently created from the command palette "Run Just Recipe" -> "new-experiment" or from the console.

```sh
just new-experiment <name>
```
