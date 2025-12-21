"""Download model and run a quick test."""

import torch
from core import ModelConfig, Qwen3Model, Qwen3Tokenizer
from core.model import KVCache
from inference import TextGenerator, GreedySampler
from utils import setup_device, download_qwen3_model, get_logger

def main():
    logger = get_logger()

    print("\n" + "=" * 70)
    print("TÉLÉCHARGEMENT ET TEST DU MODÈLE QWEN3-0.6B")
    print("=" * 70 + "\n")

    # Setup device
    device = setup_device(optimize_for_device=True)
    logger.info(f"Device: {device}")

    # Download model (base model is smaller and faster)
    print("\n1. Téléchargement du modèle et tokenizer...")
    print("-" * 70)
    paths = download_qwen3_model(
        model_type="base",
        size="0.6B",
        out_dir="./models"
    )

    print(f"\n✓ Fichiers téléchargés:")
    print(f"  - Modèle: {paths['model']}")
    print(f"  - Tokenizer: {paths['tokenizer']}")

    # Load tokenizer
    print("\n2. Chargement du tokenizer...")
    print("-" * 70)
    tokenizer = Qwen3Tokenizer(tokenizer_file_path=str(paths["tokenizer"]))
    print(f"✓ Tokenizer chargé")
    print(f"  - EOS token: {tokenizer.eos_token}")
    print(f"  - EOS token ID: {tokenizer.eos_token_id}")

    # Create and load model
    print("\n3. Création et chargement du modèle...")
    print("-" * 70)
    config = ModelConfig.qwen3_0_6b()
    print(f"Configuration:")
    print(f"  - Layers: {config.n_layers}")
    print(f"  - Embedding dim: {config.emb_dim}")
    print(f"  - Heads: {config.n_heads}")
    print(f"  - Context length: {config.context_length:,}")

    model = Qwen3Model(config)
    state_dict = torch.load(paths["model"], map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"✓ Modèle chargé sur {device}")

    # Create generator
    print("\n4. Test de génération de texte...")
    print("-" * 70)
    generator = TextGenerator(
        model,
        tokenizer,
        device,
        sampler=GreedySampler()
    )

    # Test prompts
    prompts = [
        "What is 2+2?",
        "The capital of France is",
        "Once upon a time",
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\nTest {i}/{len(prompts)}:")
        print(f"Prompt: \"{prompt}\"")
        print("Génération: ", end="", flush=True)

        # Generate with streaming
        for token in generator.generate_stream(
            prompt,
            max_new_tokens=30,
            use_cache=True,
            stop_at_eos=True
        ):
            print(token, end="", flush=True)
        print()

    print("\n" + "=" * 70)
    print("✓ TEST RÉUSSI! Le modèle fonctionne correctement.")
    print("=" * 70 + "\n")

    print("Pour plus de tests:")
    print("  python main.py generate \"Your question here\" --stream")
    print()

if __name__ == "__main__":
    main()
