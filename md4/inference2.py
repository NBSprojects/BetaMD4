# md4/inference.py

import jax
import jax.numpy as jnp
from flax import serialization
from ml_collections import config_dict
import numpy as np
import os
import flax

# --- Importations des composants originaux du dépôt ---
from md4.models import utils as model_utils
from md4.configs.md4 import text8 as text8_config_lib
from md4.input_pipeline import Text8Tokenizer
from md4 import sampling
from md4 import utils as md4_utils

# Définition d'un conteneur simple et direct pour l'inférence.
# Il doit juste avoir les attributs .params et .state pour être compatible.
@flax.struct.dataclass
class InferenceState:
    params: dict
    state: dict

def load_and_prepare_model_for_inference(config, params_path):
    """
    La méthode la plus robuste :
    1. Initialise le modèle original pour obtenir sa structure.
    2. Charge les poids EMA depuis le fichier.
    3. Construit un état d'inférence où les poids EMA sont placés dans le champ `params`.
    """
    model = model_utils.get_model(config)

    # Initialiser le modèle pour obtenir la structure des `params` et du `state`.
    rng = jax.random.PRNGKey(0)
    dummy_x = jnp.ones((1, config.data_shape[0]), dtype=jnp.int32)
    
    print("Initialisation du modèle pour obtenir la structure...")
    variables = model.init(rng, dummy_x, train=False)
    
    params_struct = variables.pop('params')
    model_state_struct = variables

    # Charger les poids EMA depuis le fichier.
    print(f"Chargement des poids EMA depuis : {params_path}")
    if not os.path.exists(params_path):
        print(f"ERREUR: Fichier de poids non trouvé : {params_path}")
        return None, None
        
    with open(params_path, 'rb') as f:
        params_bytes = f.read()
    
    loaded_ema_params = serialization.from_bytes(params_struct, params_bytes)
    print("Poids chargés avec succès.")

    # Étape cruciale : On crée un état d'inférence où le champ `params` contient les poids EMA.
    # C'est ce que la fonction de sampling utilisera.
    inference_state = InferenceState(
        params=loaded_ema_params,
        state=model_state_struct
    )

    return model, inference_state

def main():
    # --- 1. Configuration ---
    # La configuration doit correspondre PARFAITEMENT à celle utilisée pour l'entraînement.
    config = text8_config_lib.get_config()
    config.feature_dim = 64
    config.n_layers = 8
    config.num_heads = 6
    config.vocab_size = 27
    config.outside_embed = True
    config.noise_schedule = 'linear'
    config.time_features = 't'
    config.cont_time = True
    config.cond_type = 'adaln_zero'
    config.mlp_type = 'glu'
    config.sampler = 'ancestral'
    config.sampling_grid = 'cosine'

    # --- 2. Chargement du modèle ---
    params_filename = "md4_text8_step_50000.msgpack"
    path_to_m1_weights = os.path.join("trained_models", params_filename)

    model, inference_state = load_and_prepare_model_for_inference(config, path_to_m1_weights)
    if model is None:
        return

    # --- 3. Exécution de l'échantillonnage ---
    print("\n--- Lancement de l'échantillonnage avec la méthode du dépôt ---")
    
    num_samples = 8
    rng = jax.random.PRNGKey(42) # Seed fixe pour la reproductibilité

    # Compiler la fonction de sampling pour l'accélérer.
    @jax.jit
    def run_original_sampling(state):
        return sampling.simple_generate(
            rng=rng,
            train_state=state, # Notre état d'inférence est compatible
            batch_size=num_samples,
            model=model,
            conditioning=None
        )

    print("Démarrage de la génération des échantillons...")
    sampled_tokens = run_original_sampling(inference_state)
    print("Génération terminée.")
    
    # --- 4. Affichage des résultats ---
    tokenizer = Text8Tokenizer()
    detokenized_texts = md4_utils.detokenize_texts(sampled_tokens, tokenizer)

    for i, text in enumerate(detokenized_texts):
        print(f"\n--- Échantillon {i+1} ---")
        print(text)

if __name__ == '__main__':
    main()