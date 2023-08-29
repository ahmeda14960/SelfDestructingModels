
from collections import defaultdict
import os
from omegaconf import OmegaConf
import json
import seaborn as sns
import pandas as pd

mydir = "outputs/None/bios_repro__1.0__1.0__0.0__16__2023-08-28_20-08-47__51880819"
_aggregated_data = defaultdict(list)

skip_tasks = ["regression"]

graph = "best"
if graph == "best":
    models = [("pretrained" , {}, 'BERT'), 
            ('random', {}, 'Random'),
            ('loaded', {'l_bad_adapted' : '0.0',
                        'l_linear_mi' : '0.0',
                        'l_bad_adapted_grad' : '0.0',
                        'max_adapt_steps' : '0',
                        'train_steps' : '5000'}, 'BERT (tuned, professions)'),
            ('loaded', {'l_bad_adapted' : '0.0',
                        'l_linear_mi' : '1.0',
                        'l_bad_adapted_grad' : '0.0',
                        'max_adapt_steps' : '16'}, 'MLAC'),
            ('loaded', {'l_bad_adapted' : '1.0',
                'l_linear_mi' : '0.0',
                'l_bad_adapted_grad' : '0.0',
                'max_adapt_steps' : '0'}, 'AC')
                        ]
elif graph == "steps":
    models = [("pretrained" , {}, 'BERT'), 
            ('random', {}, 'Random'),
            ('loaded', {
                        'l_bad_adapted_grad' : '0.0',
                        'max_adapt_steps' : '16'}, '16 steps'),
            ('loaded', {
                'l_bad_adapted_grad' : '0.0',
                'max_adapt_steps' : '4'}, '4 steps'),
            ('loaded', {
                'l_bad_adapted_grad' : '0.0',
                'max_adapt_steps' : '0'}, '0 steps')
                        ]
elif graph == "grad":
    models = [("pretrained" , {}, 'BERT'), 
            ('random', {}, 'Random'),
            ('loaded', {'l_bad_adapted_grad' : '0.0', 'max_adapt_steps' : '16'}, 'No Grad Penalty'),
            ('loaded', {
                'l_bad_adapted_grad' : '1.0', 'max_adapt_steps' : '16'}, 'Grad Penalty')
                        ]
elif graph == "mi":
    models = [("pretrained" , {}, 'BERT'), 
        ('random', {}, 'Random'),
        ('loaded', {'l_linear_mi' : '0.0', 'l_bad_adapted_grad' : '0.0'}, 'No head adjustment'),
        ('loaded', {
            'l_linear_mi' : '1.0', 'l_bad_adapted_grad' : '0.0'}, 'Head adjustment')
                    ]

for dirpath, dirnames, filenames in os.walk(mydir):
    import ipdb; ipdb.set_trace()
    if "eval_info.json" in filenames:
        # We're in a result dir
        overrides = OmegaConf.load(os.path.join(dirpath, ".hydra/overrides.yaml"))
        _override_dict = {}
        # get seed, eval type and experiment
        for override in overrides:
            k, v = override.split("=")
            _override_dict[k] = v

        if _override_dict["experiment"] in skip_tasks:
            continue
        eval_type = _override_dict["eval_network_type"]

        # if the eval type isn't relevant to graph
        # continue
        if eval_type not in [x[0] for x in models]:
            continue

        # load model trained weights if necessary
        # otherwise it's a pretrained / random model
        if eval_type == "loaded":
            overrides = OmegaConf.load(os.path.join(dirpath, "loaded_model_conf.yaml"))
            _loaded_model_override_dict = {}
            for override in overrides:
                k, v = override.split("=")
                _loaded_model_override_dict[k] = v
            skip_model = True
            for t, v, name in models:
                if t != "loaded":
                    continue
                all_true = True
                for key, value in v.items():
                    if key not in _loaded_model_override_dict or _loaded_model_override_dict[key] != value:
                        all_true = False
                if all_true:
                    model_name = name
                    skip_model = False
            if skip_model:
                continue
        else:
            model_name = models[[x[0] for x in models].index(eval_type)][2]

        _aggregated_data["Dataset Size"].append(int(_override_dict["adversary.n_examples"]))
        _aggregated_data["experiment"].append(_override_dict["experiment"])
        _aggregated_data["seed"].append(_override_dict["seed"])
        _aggregated_data["Model"].append(model_name)
        with open(os.path.join(dirpath, "eval_info.json")) as f:
            _results = json.load(f)
        if "eval_only_bad" in _override_dict and _override_dict["eval_only_bad"] == "True":
            acc_key = "acc" # TODO: later fix so we append the bad modified again
        else:
            acc_key = "acc_eval_bad"
        if _override_dict["experiment"] == "regression":
            solvesteps_key = "ybad_test_solve_datapoints/0.7"
        else:
            solvesteps_key = "genders_test_solve_datapoints/0.7"
        try:
            _aggregated_data["solve_datapoints"].append(_results[solvesteps_key])
        except:
            # import pdb; pdb.set_trace()
            _aggregated_data["solve_datapoints"].append(_results[solvesteps_key + "_eval_bad"])
        _aggregated_data["Gender Accuracy (Post-adaptation)"].append(_results[acc_key])

df = pd.DataFrame.from_dict(_aggregated_data)


cmap = sns.color_palette("colorblind", len(df["Model"].unique()))


sns.relplot(
    data=df, x="Dataset Size", y="Gender Accuracy (Post-adaptation)", col="experiment",
    hue="Model", style="Model", kind="line", palette=cmap
).figure.savefig("out.png")

sns.relplot(
    data=df, x="Dataset Size", y="solve_datapoints", col="experiment",
    hue="Model", style="Model", kind="line", palette=cmap
).figure.savefig("out_datapoints.png")

    # if not dirnames:
    #     print (dirpath, "has 0 subdirectories and", len(filenames), "files")
