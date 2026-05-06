import torch
import random
import difflib
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


def overlap(seq1, seq2):
    """Finds the overlap between two token ID sequences."""
    seq1 = [str(x) for x in seq1.tolist()]
    seq2 = [str(x) for x in seq2.tolist()]
    matching_tokens1, matching_tokens2 = [], []
    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    for op in matcher.get_opcodes():
        if op[0] == "equal":
            matching_tokens1 += list(range(op[1], op[2]))
            matching_tokens2 += list(range(op[3], op[4]))
    return matching_tokens1, matching_tokens2


def prob_next_token(matrix, token_ids, next_idx, lm):
    """Calculate log-probability of the next token at position next_idx."""
    log_softmax = lm["log_softmax"]
    next_token_scores = matrix[next_idx]
    target_word_id = token_ids[0][next_idx]
    log_prob = log_softmax(next_token_scores)[target_word_id]
    return {"log_prob": log_prob}


def compare_pair(pair, lm):
    """Compare a pair of sentences based on pseudo-log-likelihood of matching tokens."""
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    device = lm.get("device", "cpu")

    sent1, sent2 = pair["sent1"], pair["sent2"]

    # Tokenize with and without BOS; always move tensors to device
    sent1_token_ids = tokenizer.encode(
        tokenizer.bos_token + sent1, return_tensors="pt", add_special_tokens=False
    ).to(device)
    sent2_token_ids = tokenizer.encode(
        tokenizer.bos_token + sent2, return_tensors="pt", add_special_tokens=False
    ).to(device)
    sent1_token_ids_no_bos = tokenizer.encode(
        sent1, return_tensors="pt", add_special_tokens=False
    ).to(device)
    sent2_token_ids_no_bos = tokenizer.encode(
        sent2, return_tensors="pt", add_special_tokens=False
    ).to(device)

    matching_tokens1, matching_tokens2 = overlap(
        sent1_token_ids_no_bos[0].cpu(), sent2_token_ids_no_bos[0].cpu()
    )
    matching_tokens = tokenizer.convert_ids_to_tokens(
        sent1_token_ids_no_bos[0][matching_tokens1].cpu()
    )

    all_tokens1 = tokenizer.convert_ids_to_tokens(sent1_token_ids_no_bos[0])
    all_tokens2 = tokenizer.convert_ids_to_tokens(sent2_token_ids_no_bos[0])

    non_matching_tokens1 = [
        tok for i, tok in enumerate(all_tokens1) if i not in matching_tokens1
    ]

    non_matching_tokens2 = [
        tok for i, tok in enumerate(all_tokens2) if i not in matching_tokens2
    ]

    with torch.no_grad():
        # output-matrix from model for both sentences [seq_len, vocab_size]
        matrix1 = model(sent1_token_ids)[0].squeeze(0)
        matrix2 = model(sent2_token_ids)[0].squeeze(0)

    sent1_log_probs = sum(
        prob_next_token(matrix1, sent1_token_ids_no_bos, idx, lm)["log_prob"].item()
        for idx in matching_tokens1
    )
    sent2_log_probs = sum(
        prob_next_token(matrix2, sent2_token_ids_no_bos, idx, lm)["log_prob"].item()
        for idx in matching_tokens2
    )

    preferred = "A" if sent1_log_probs > sent2_log_probs else "B" if sent2_log_probs > sent1_log_probs else "Equal"
    
    return {
        "matching_tokens": matching_tokens,
        "sent1_pseudolog": sent1_log_probs,
        "sent2_pseudolog": sent2_log_probs,
        "preferred": preferred,
        "overlap": (non_matching_tokens1, non_matching_tokens2)
    }


def parse_answer(answer):
    """
    Parse the model's answer to extract a preference ('1' or '2').
    Matches digits 1/2 as well as Dutch ordinal words 'eerste'/'tweede'.
    Returns '1', '2', or None if ambiguous.
    """
    import re
    answer_lower = answer.lower()

    # Check for ordinal words first
    has_eerste = bool(re.search(r'\beerste\b', answer_lower))
    has_tweede = bool(re.search(r'\btweede\b', answer_lower))
    if has_eerste and not has_tweede:
        return '1'
    if has_tweede and not has_eerste:
        return '2'

    # Find the first standalone digit 1 or 2
    match = re.search(r'\b([12])\b', answer_lower)
    if match:
        # Make sure the other digit doesn't also appear as a standalone number
        other = '2' if match.group(1) == '1' else '1'
        if not re.search(r'\b' + other + r'\b', answer_lower):
            return match.group(1)

    return None


def compare_pair_prompt_batch(entries, lm, template_id=1):
    """Evaluate multiple sentence pairs in a batch with prompt-based evaluation.
    
    template_id selects the prompt template:
      1 - Probability framing  ('waarschijnlijker'), labels 1/2, concise
      2 - Naturalness framing  ('natuurlijker om te horen'), labels 1/2, concise
      3 - Plausibility framing ('aannemelijker'), labels 1/2, concise
    """
    tokenizer = lm["tokenizer"]
    model = lm["model"]
    device = lm["device"]

    prompts = []
    swaps = []

    for entry in entries:
        s1, s2 = entry["sent1"], entry["sent2"]
        if random.random() > 0.5:
            s1, s2 = s2, s1
            swapped = True
        else:
            swapped = False

        if template_id == 1:
            prompt = (
                "Welke zin is waarschijnlijker om van een mens gehoord te hebben? "
                "Antwoord met alleen '1' of '2', niets anders.\n\n"
                f"Zin 1: {s1}\n"
                f"Zin 2: {s2}\n"
                "Antwoord:"
            )
        elif template_id == 2:
            prompt = (
                "Welke zin klinkt het meest natuurlijk om van een mens te horen? "
                "Antwoord met alleen '1' of '2', niets anders.\n\n"
                f"Zin 1: {s1}\n"
                f"Zin 2: {s2}\n"
                "Antwoord:"
            )
        elif template_id == 3:
            prompt = (
                "Welke zin is het meest aannemelijk als iets wat een mens zou zeggen? "
                "Antwoord met alleen '1' of '2', niets anders.\n\n"
                f"Zin 1: {s1}\n"
                f"Zin 2: {s2}\n"
                "Antwoord:"
            )
        else:
            raise ValueError(f"Unknown template_id: {template_id}. Choose 1, 2, or 3.")

        prompts.append(prompt)
        swaps.append(swapped)

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,  # greedy decoding — fully deterministic
            pad_token_id=tokenizer.pad_token_id,
        )

    results = []
    for i in range(len(entries)):
        start = inputs["input_ids"].shape[1]
        gen_ids = outputs[i][start:]
        answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        parsed = parse_answer(answer)
        if parsed == '1':
            preferred = "A" if not swaps[i] else "B"
        elif parsed == '2':
            preferred = "B" if not swaps[i] else "A"
        else:
            preferred = "Equal"

        results.append({
            "preferred": preferred,
            "raw_output": answer,
            "swapped": swaps[i]
        })

    return results

def set_seed(seed=42):
    """Set the seed for random operations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # For GPU support
    torch.cuda.manual_seed_all(seed)  # For all GPU devices
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable auto-tuning to make the results reproducible
    print(f"[INFO] Seed set to {seed}")