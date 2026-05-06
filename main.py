import argparse
import torch
import pandas as pd
from code.models import load_model, prepare_lm
from code.eval_likelihood import evaluate_likelihood
from code.eval_prompt import evaluate_prompt
from code.utils import set_seed
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--data_file", type=str, default="preprocessed_final.csv")  # Data file in the data folder
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--eval_mode", type=str, default="likelihood", choices=["likelihood", "prompt"])
    parser.add_argument("--output_dir", type=str, default="experiment_results")
    args = parser.parse_args()

    set_seed(42)

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Define path to the data folder
    data_folder = os.path.join(os.path.dirname(__file__), 'data')

    # Check if the data file exists in the folder
    data_file_path = os.path.join(data_folder, args.data_file)

    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"[ERROR] The file at {data_file_path} was not found. Please preprocess your data first.")

    # Load and preprocess data
    data = pd.read_csv(data_file_path, sep="\t")

    # Load model and tokenizer
    tokenizer, model = load_model(args.model_name, device)
    lm = prepare_lm(model, tokenizer, device)

    # Run the appropriate evaluation
    if args.eval_mode == "likelihood":
        evaluate_likelihood(lm, data, sample_size=args.sample_size, model_name=args.model_name)
    elif args.eval_mode == "prompt":
        _, template_scores, mean_score, std_score = evaluate_prompt(
            lm, data, sample_size=args.sample_size, model_name=args.model_name, output_dir=args.output_dir
        )
        print("\n========== PROMPT EVALUATION RESULTS ==========")
        for template_id, score in template_scores.items():
            from code.eval_prompt import TEMPLATES
            print(f"  T{template_id} ({TEMPLATES[template_id]}): {score}%")
        print(f"  Mean stereotype score: {mean_score}%  (std: {std_score}%)")
        print("================================================")

if __name__ == "__main__":
    main()
