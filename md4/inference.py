# md4/inference.py

import jax
import jax.numpy as jnp
from flax import serialization
from ml_collections import config_dict
import numpy as np
import os
import flax
from flax.struct import dataclass
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from scipy import stats
from typing import Any

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Supprimer les avertissements XLA

# --- Importations des composants originaux du dépôt ---
from md4.models import utils as model_utils
from md4.configs.md4 import text8 as text8_config_lib
from md4.input_pipeline import Text8Tokenizer
from md4 import sampling
from md4 import utils as md4_utils

# Définition d'un conteneur simple pour l'inférence, utilisant flax.struct.dataclass.
# Il contient UNIQUEMENT ce dont la fonction de sampling a besoin.
@dataclass
class InferenceState:
    params: dict
    state: Any


def load_and_prepare_model_for_inference(config, params_path):
    """
    Initialise le modèle, charge les poids, et retourne notre état d'inférence simple.
    """
    model = model_utils.get_model(config)

    # Initialiser le modèle pour obtenir la structure complète des variables.
    rng = jax.random.PRNGKey(0)
    dummy_x = jnp.ones((1, config.data_shape[0]), dtype=jnp.int32)
    
    print("Initialisation du modèle pour obtenir la structure...")
    variables = model.init(rng, dummy_x, train=False)
    
    params_struct = variables.pop('params')
    # Le reste des variables constitue l'état du modèle (ex: stats de batchnorm)
    model_state_struct = variables

    # Charger les poids depuis le fichier.
    print(f"Chargement des poids depuis : {params_path}")
    if not os.path.exists(params_path):
        print(f"ERREUR: Fichier de poids non trouvé : {params_path}")
        return None, None
        
    with open(params_path, 'rb') as f:
        params_bytes = f.read()
    
    loaded_params = serialization.from_bytes(params_struct, params_bytes)
    print("Poids chargés avec succès.")

    # Création directe de notre conteneur d'inférence.
    # Il n'y a pas de méthode .create(), on l'instancie comme une classe normale.
    inference_state = InferenceState(
        params=loaded_params,
        state=model_state_struct
    )

    return model, inference_state

def main():
    # --- 1. Configuration ---
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
    config.timesteps = 1000

    # --- 2. Chargement du modèle ---
    params_filename = "md4_text8_step_70000.msgpack"
    path_to_m1_weights = os.path.join("trained_models", params_filename)

    model, inference_state = load_and_prepare_model_for_inference(config, path_to_m1_weights)
    if model is None:
        return

    # --- 3. Exécution de l'échantillonnage avec 1000 samples ---
    print("\n--- Lancement de l'échantillonnage avec 1000 samples ---")
    
    num_samples = 10  # Modifié de 8 à 1000
    rng = jax.random.PRNGKey(42)

    @jax.jit
    def run_original_sampling(state):
        # La fonction `simple_generate` est compatible avec notre `InferenceState`
        # car elle ne requiert que les attributs `.params` et `.state`.
        return sampling.simple_generate(
            rng=rng,
            train_state=state,
            batch_size=num_samples,
            model=model,
            conditioning=None
        )

    print("Démarrage de la génération des échantillons...")
    sampled_tokens = run_original_sampling(inference_state)
    print("Génération terminée.")


    # --- 4. Afficher quelques échantillons du modèle ---
    print("\n--- Exemples d'échantillons générés par le modèle ---")
    tokenizer = Text8Tokenizer()
    detokenized_texts = md4_utils.detokenize_texts(sampled_tokens[:5], tokenizer)  # Afficher seulement 5 exemples

    for i, text in enumerate(detokenized_texts):
        print(f"\n--- Échantillon {i+1} ---")
        print(text)

    # --- 5. Analyse de phrases masquées ---
    print("\n--- Lancement de l'analyse de phrases masquées ---")
    
    # Phrases anglaises bien formées
    sentences = [
        "the quick brown fox jumps over the lazy dog",
        "a stitch in time saves nine",
        "curiosity killed the cat",
        "an apple a day keeps the doctor away",
    ]
    
    # MASK_TOKEN_ID est vocab_size, qui est 27 dans notre config
    MASK_TOKEN_ID = config.vocab_size 

    def mask_sentence(sentence_tokens, mask_fraction=0.5, rng_key=None):
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        
        num_tokens = len(sentence_tokens)
        num_to_mask = int(num_tokens * mask_fraction)
        
        # Choisir aléatoirement des indices à masquer
        indices_to_mask = jax.random.choice(rng_key, num_tokens, (num_to_mask,), replace=False)
        
        # Créer le tenseur masqué
        masked_tokens = jnp.array(sentence_tokens)
        masked_tokens = masked_tokens.at[indices_to_mask].set(MASK_TOKEN_ID)
        
        return masked_tokens, indices_to_mask

    @jax.jit
    def run_inpainting_analysis(state, masked_tokens, t):
        # La méthode 'predict_x' nous donne les logits pour chaque token
        logits, _ = model.apply(
            {'params': state.params, **state.state},
            masked_tokens,
            t=jnp.array(t),
            cond=None,
            train=False,
            method=model.predict_x
        )
        return logits

    # Analyser chaque phrase
    analysis_rng = jax.random.PRNGKey(123)
    for sentence in sentences:
        print(f"\n--- Analyse de la phrase : '{sentence}' ---")
        
        # Tokenize
        original_tokens = tokenizer.encode(sentence.encode('utf-8'))
        
        # Masquage
        analysis_rng, mask_rng = jax.random.split(analysis_rng)
        masked_tokens, masked_indices = mask_sentence(original_tokens, mask_fraction=0.15, rng_key=mask_rng)

        # Préparation pour l'affichage
        masked_sentence_list = list(tokenizer.decode(masked_tokens))
        for idx in masked_indices:
            masked_sentence_list[idx] = '_' # Mettre un caractère visible pour le masque
        print(f"Phrase masquée : {''.join(masked_sentence_list)}")

        # Obtenir les prédictions du modèle
        # Nous utilisons un petit 't' pour être proche des données réelles
        # Le modèle a 1000 timesteps, donc t=1/1000 est le premier pas.
        t_val = 1. / float(config.timesteps)
        
        # Ajout d'une dimension batch
        logits = run_inpainting_analysis(inference_state, masked_tokens[None, :], t_val)
        logits = logits[0] # Retirer la dimension batch

        # Calculer l'argmax et l'entropie pour les positions masquées
        predictions = jnp.argmax(logits, axis=-1)
        
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        probs = jnp.exp(log_probs)
        entropy = -jnp.sum(probs * log_probs, axis=-1)

        print("Prédictions pour les tokens masqués :")
        print("Indice | Original | Prédit | Entropie")
        print("---------------------------------------")
        
        original_chars = list(sentence)
        predicted_chars = list(tokenizer.decode(predictions))

        for i in sorted(masked_indices.tolist()):
            original_char = original_chars[i]
            predicted_char = predicted_chars[i]
            token_entropy = entropy[i]
            
            print(f"{i: <6} | {original_char: <8} | {predicted_char: <6} | {token_entropy:.4f}")


if __name__ == '__main__':
    main()