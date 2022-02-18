import argparse
import torch
import re

from model import Transformer
from tokenizers import Tokenizer


def _parse_args() -> object:
    parser = argparse.ArgumentParser(
        description='Forocoches inference parser')

    parser.add_argument('--load_model', type=str, default="models/fcgen_epoch_3_v.pth",
                        help='model to load and do inference')
    parser.add_argument('--prompt', type=str,
                        help='prompt for the model')
    parser.add_argument('--tokenizer_config', type=str, default="models/fctokenizer-small/config.json",
                        help='location of the tokenizer config file')

    args = parser.parse_args()
    return args


def main(args) -> None:
    tokenizer = Tokenizer.from_file(args.tokenizer_config)

    #TODO: Create a configuration file for the Hyperparameters
    N_POSITIONS = 500
    model = Transformer(tokenizer,
                        num_tokens=tokenizer.get_vocab_size(),
                        dim_model=768,
                        d_hid=3072,
                        num_heads=12,
                        num_layers=12,
                        dropout_p=0.1,
                        n_positions=N_POSITIONS,
                        )
    model.load_state_dict(torch.load(args.load_model))

    result = model.predict(args.prompt, temperature=0.7, top_p=0.92, max_length=200)
    result = result.replace("<|THREAD|>", "").replace("<|BODY|>", "\n\n").replace("<|COMMENT|>", "\n\n--- COMMENT ---\n")
    result = re.sub(r'(\/\/.+\/(\w+)\.\w+)', r':\g<2>:', result)

    with open("prediction.txt", "w", encoding="utf-8") as f:
        f.write(result)

    print("OUT:", result)


if __name__ == "__main__":
    args = _parse_args()
    main(args)
