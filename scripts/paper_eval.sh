if (( "$#" == 0 )); then 
    ID=$(python -c "import random; chars='qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890'; print(''.join([random.choice(chars) for _ in range(7)]))")
else
    ID=$1
fi

echo "Running batch of jobs with id $ID"

ebatch model_eval_1      slconf_jag "python train.py -m hydra/launcher=nlp_joblib experiment=regression eval_only=True eval_network_type=loaded adversary.n_examples=20,50,100,200 seed=0,1,2,3,4,5 +eval_loaded_model_dir=/u/scr/nlp/data/sd_models/H63mlVF/regression__0.0__1.0__0.0__0__2022-05-19_11-27-01__4260118/ eval_only_bad=True batch_hash=$ID"
ebatch model_eval_2      slconf_jag "python train.py -m hydra/launcher=nlp_joblib experiment=regression eval_only=True eval_network_type=loaded adversary.n_examples=20,50,100,200 seed=0,1,2,3,4,5 +eval_loaded_model_dir=/u/scr/nlp/data/sd_models/H63mlVF/regression__0.0__1.0__0.0__16__2022-05-19_11-27-01__307662/ eval_only_bad=True batch_hash=$ID"
ebatch model_eval_3      slconf_jag "python train.py -m hydra/launcher=nlp_joblib experiment=regression eval_only=True eval_network_type=loaded adversary.n_examples=20,50,100,200 seed=0,1,2,3,4,5 +eval_loaded_model_dir=/u/scr/nlp/data/sd_models/H63mlVF/regression__0.0__1.0__0.0__4__2022-05-19_11-27-01__93729207/ eval_only_bad=True batch_hash=$ID"
ebatch model_eval_4      slconf_jag "python train.py -m hydra/launcher=nlp_joblib experiment=regression eval_only=True eval_network_type=loaded adversary.n_examples=20,50,100,200 seed=0,1,2,3,4,5 +eval_loaded_model_dir=/u/scr/nlp/data/sd_models/H63mlVF/regression__0.0__1.0__1.0__16__2022-05-19_11-27-04__88034090/ eval_only_bad=True batch_hash=$ID"
ebatch model_eval_5      slconf_jag "python train.py -m hydra/launcher=nlp_joblib experiment=regression eval_only=True eval_network_type=loaded adversary.n_examples=20,50,100,200 seed=0,1,2,3,4,5 +eval_loaded_model_dir=/u/scr/nlp/data/sd_models/H63mlVF/regression__0.0__1.0__1.0__4__2022-05-19_11-27-04__81595274/ eval_only_bad=True batch_hash=$ID"
ebatch model_eval_6      slconf_jag "python train.py -m hydra/launcher=nlp_joblib experiment=regression eval_only=True eval_network_type=loaded adversary.n_examples=20,50,100,200 seed=0,1,2,3,4,5 +eval_loaded_model_dir=/u/scr/nlp/data/sd_models/H63mlVF/regression__1.0__0.0__0.0__0__2022-05-19_11-27-02__70897152/ eval_only_bad=True batch_hash=$ID"
ebatch model_eval_7      slconf_jag "python train.py -m hydra/launcher=nlp_joblib experiment=regression eval_only=True eval_network_type=loaded adversary.n_examples=20,50,100,200 seed=0,1,2,3,4,5 +eval_loaded_model_dir=/u/scr/nlp/data/sd_models/H63mlVF/regression__1.0__0.0__0.0__16__2022-05-19_11-26-58__41461981/ eval_only_bad=True batch_hash=$ID"
ebatch model_eval_8      slconf_jag "python train.py -m hydra/launcher=nlp_joblib experiment=regression eval_only=True eval_network_type=loaded adversary.n_examples=20,50,100,200 seed=0,1,2,3,4,5 +eval_loaded_model_dir=/u/scr/nlp/data/sd_models/H63mlVF/regression__1.0__0.0__0.0__4__2022-05-19_11-26-58__43065949/ eval_only_bad=True batch_hash=$ID"
ebatch model_eval_9      slconf_jag "python train.py -m hydra/launcher=nlp_joblib experiment=regression eval_only=True eval_network_type=loaded adversary.n_examples=20,50,100,200 seed=0,1,2,3,4,5 +eval_loaded_model_dir=/u/scr/nlp/data/sd_models/H63mlVF/regression__1.0__0.0__1.0__0__2022-05-19_11-26-58__8601468/ eval_only_bad=True batch_hash=$ID"
ebatch model_eval_10      slconf_jag "python train.py -m hydra/launcher=nlp_joblib experiment=regression eval_only=True eval_network_type=loaded adversary.n_examples=20,50,100,200 seed=0,1,2,3,4,5 +eval_loaded_model_dir=/u/scr/nlp/data/sd_models/H63mlVF/regression__1.0__0.0__1.0__16__2022-05-19_11-26-58__4720976/ eval_only_bad=True batch_hash=$ID"
ebatch model_eval_11      slconf_jag "python train.py -m hydra/launcher=nlp_joblib experiment=regression eval_only=True eval_network_type=loaded adversary.n_examples=20,50,100,200 seed=0,1,2,3,4,5 +eval_loaded_model_dir=/u/scr/nlp/data/sd_models/H63mlVF/regression__1.0__0.0__1.0__4__2022-05-19_11-26-58__15485531/ eval_only_bad=True batch_hash=$ID"

# nlprun command
nlprun -a sdm -n sdm_sweep -w /iris/u/ahmedah/SelfDestructingModels -o /iris/u/ahmedah/SelfDestructingModels/slurm.out -p high 'python3 -m train experiment=bios_repro eval_only=True eval_network_type=random seed=0'
'python -m train --multirun experiment=bios_repro eval_only=True adversary.n_examples=20,50,100,200 eval_network_type=random seed=0'
# AC
python -m train --multirun experiment=bios_repro eval_only=True adversary.n_examples=20,50,100,200 eval_network_type=loaded  +eval_loaded_model_dir=/lfs/ampere1/0/ahmedah/SelfDestructingModels/outputs/None/bios_ac__1.0__1.0__0.0__0__2023-08-27_18-29-49__19879353/ seed=0
# MLAC
python -m train --multirun experiment=bios_repro eval_only=True adversary.n_examples=20,50,100,200 eval_network_type=loaded  +eval_loaded_model_dir=/lfs/ampere1/0/ahmedah/SelfDestructingModels/outputs/None/bios_repro__1.0__1.0__0.0__16__2023-08-27_21-29-09__87983824/ seed=0