def calculate_bigram_statistics(sequences, vocab_size=27):
    """
    Calcule les statistiques de 2-grammes pour une liste de séquences.
    
    Args:
        sequences: Liste de séquences (arrays numpy ou listes)
        vocab_size: Taille du vocabulaire
    
    Returns:
        dict: Dictionnaire contenant les statistiques de 2-grammes
    """
    bigram_counts = defaultdict(int)
    total_bigrams = 0
    
    for seq in sequences:
        for i in range(len(seq) - 1):
            bigram = (int(seq[i]), int(seq[i + 1]))
            bigram_counts[bigram] += 1
            total_bigrams += 1
    
    # Calculer les probabilités
    bigram_probs = {}
    for bigram, count in bigram_counts.items():
        bigram_probs[bigram] = count / total_bigrams
    
    # Créer une matrice de transition
    transition_matrix = np.zeros((vocab_size, vocab_size))
    for (i, j), prob in bigram_probs.items():
        transition_matrix[i, j] = prob
    
    return {
        'bigram_counts': dict(bigram_counts),
        'bigram_probs': bigram_probs,
        'transition_matrix': transition_matrix,
        'total_bigrams': total_bigrams,
        'unique_bigrams': len(bigram_counts)
    }

def generate_uniform_sequences(num_samples, seq_length, vocab_size=27, rng_seed=123):
    """
    Génère des séquences uniformes aléatoires pour comparaison.
    
    Args:
        num_samples: Nombre de séquences à générer
        seq_length: Longueur de chaque séquence
        vocab_size: Taille du vocabulaire
        rng_seed: Graine pour la génération aléatoire
    
    Returns:
        Liste de séquences uniformes
    """
    np.random.seed(rng_seed)
    uniform_sequences = []
    
    for _ in range(num_samples):
        seq = np.random.randint(0, vocab_size, seq_length)
        uniform_sequences.append(seq)
    
    return uniform_sequences

def compare_bigram_statistics(model_stats, uniform_stats):
    """
    Compare les statistiques de 2-grammes entre le modèle et les séquences uniformes.
    
    Args:
        model_stats: Statistiques du modèle
        uniform_stats: Statistiques des séquences uniformes
    
    Returns:
        dict: Résultats de la comparaison
    """
    model_matrix = model_stats['transition_matrix']
    uniform_matrix = uniform_stats['transition_matrix']
    
    # Test de Chi-carré
    # On aplatie les matrices et supprime les valeurs nulles
    model_flat = model_matrix.flatten()
    uniform_flat = uniform_matrix.flatten()
    
    # Garder seulement les positions où au moins une des deux matrices est non-nulle
    non_zero_mask = (model_flat > 0) | (uniform_flat > 0)
    model_filtered = model_flat[non_zero_mask]
    uniform_filtered = uniform_flat[non_zero_mask]
    
    # Normaliser pour que la somme soit 1
    model_filtered = model_filtered / model_filtered.sum() if model_filtered.sum() > 0 else model_filtered
    uniform_filtered = uniform_filtered / uniform_filtered.sum() if uniform_filtered.sum() > 0 else uniform_filtered
    
    # Calculer la distance de Kullback-Leibler
    def kl_divergence(p, q):
        epsilon = 1e-10
        p = np.maximum(p, epsilon)
        q = np.maximum(q, epsilon)
        return np.sum(p * np.log(p / q))
    
    kl_model_to_uniform = kl_divergence(model_filtered, uniform_filtered)
    kl_uniform_to_model = kl_divergence(uniform_filtered, model_filtered)
    
    # Calculer l'entropie
    def entropy(probs):
        epsilon = 1e-10
        probs = np.maximum(probs, epsilon)
        return -np.sum(probs * np.log2(probs))
    
    model_entropy = entropy(model_filtered)
    uniform_entropy = entropy(uniform_filtered)
    
    # Distance de Jensen-Shannon
    m = (model_filtered + uniform_filtered) / 2
    js_divergence = (kl_divergence(model_filtered, m) + kl_divergence(uniform_filtered, m)) / 2
    
    return {
        'kl_model_to_uniform': kl_model_to_uniform,
        'kl_uniform_to_model': kl_uniform_to_model,
        'js_divergence': js_divergence,
        'model_entropy': model_entropy,
        'uniform_entropy': uniform_entropy,
        'model_unique_bigrams': model_stats['unique_bigrams'],
        'uniform_unique_bigrams': uniform_stats['unique_bigrams'],
        'model_total_bigrams': model_stats['total_bigrams'],
        'uniform_total_bigrams': uniform_stats['total_bigrams']
    }

def print_analysis_results(comparison_results, model_stats, uniform_stats):
    """
    Affiche les résultats de l'analyse statistique.
    """
    print("\n" + "="*70)
    print("ANALYSE STATISTIQUE DES 2-GRAMMES")
    print("="*70)
    
    print(f"\n📊 STATISTIQUES GÉNÉRALES:")
    print(f"  • Modèle - 2-grammes uniques: {comparison_results['model_unique_bigrams']}")
    print(f"  • Modèle - Total 2-grammes: {comparison_results['model_total_bigrams']}")
    print(f"  • Uniforme - 2-grammes uniques: {comparison_results['uniform_unique_bigrams']}")
    print(f"  • Uniforme - Total 2-grammes: {comparison_results['uniform_total_bigrams']}")
    
    print(f"\n🔍 MESURES DE DIVERGENCE:")
    print(f"  • Divergence KL (Modèle → Uniforme): {comparison_results['kl_model_to_uniform']:.4f}")
    print(f"  • Divergence KL (Uniforme → Modèle): {comparison_results['kl_uniform_to_model']:.4f}")
    print(f"  • Divergence Jensen-Shannon: {comparison_results['js_divergence']:.4f}")
    
    print(f"\n📈 ENTROPIE:")
    print(f"  • Entropie du modèle: {comparison_results['model_entropy']:.4f} bits")
    print(f"  • Entropie uniforme: {comparison_results['uniform_entropy']:.4f} bits")
    
    print(f"\n🎯 INTERPRÉTATION:")
    if comparison_results['js_divergence'] < 0.1:
        print("  • Les distributions sont très similaires (JS < 0.1)")
        print("  • Le modèle génère des séquences proches de l'aléatoire uniforme")
    elif comparison_results['js_divergence'] < 0.3:
        print("  • Les distributions sont modérément différentes (0.1 ≤ JS < 0.3)")
        print("  • Le modèle montre quelques patterns structurés")
    else:
        print("  • Les distributions sont significativement différentes (JS ≥ 0.3)")
        print("  • Le modèle a appris des structures linguistiques importantes")
    
    if comparison_results['model_entropy'] > comparison_results['uniform_entropy']:
        print("  • Le modèle est plus aléatoire que l'uniforme (entropie plus élevée)")
    elif comparison_results['model_entropy'] < comparison_results['uniform_entropy']:
        print("  • Le modèle est plus structuré que l'uniforme (entropie plus faible)")
    else:
        print("  • Le modèle et l'uniforme ont des niveaux d'entropie similaires")
