# Federated bagging and boosting examples

## Getting started

### Setup the director
To run an example, enter its folder, open a terminal window in the `director` subfolder and run:
```
fx director start --disable-tls -nn False -c director.yaml
```
to set up the director. This command will instantiate the director process on the network address and port specified in the `director.yaml` file, and the `-nn False` argument will suggest to the runtime that the federated bagging or boosting algorithm will be used.

### Setup the envoys
Then, open a different terminal window in each `envoy` folder provided by the tutorial (usually 10, in a few cases 2). These terminal windows can be on the same machine as the `director` process or on different ones. For each terminal run:
```
fx envoy start -n $ENVOY --disable-tls --envoy-config-path envoy_config.yaml -dh $DIRECTOR_ADDRESS -dp $DIRECTOR_PORT
```
to set up the envoy. The `$ENVOY` variable contains the name of the envoy (usually named as ist folder), `$DIRECTOR_ADDRESS` contains the FQDN of the `director` node, and `$DIRECTOR_PORT` the `director` listening port (these last two values are also specified in the `director.yaml` file). If fewer envoys than the maximum number are provided, the training will exploit only those instantiated.

### Start the federation
Finally, open the last terminal window in the `workspace` folder of the selected example and simply run the experiment execution file; for example, in the case of the AdaBoostF_Adult tutorial, the command is the following:
```
python AdaboostF_adult.py
```
The experiment execution file can be modified according to the user's needs; it is also possible to specify some parameters from the command line; just run the execution file with the `-h` flag to check them.

### WandB support
If you want to log the results into [WandB](https://wandb.ai/site), the name assigned to each envoy should be something like `envoy_$N`, where `$N` is the envoy rank. Then, you should set `LOG_WANDB=True` in both the `openfl-extended/openfl/component/collaborator/Collaborator.py` file (in the installation path, not the repository clone one) and in the experiment execution file.



## Experiment personalisation