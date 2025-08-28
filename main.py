import transformers as tr
import numpy as np
import torch

# testing=True prints probabilities and objective functions for each candidate token
testing = True

amateur_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
expert_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
amateur_model = tr.AutoModelForCausalLM.from_pretrained(amateur_path)
expert_model = tr.AutoModelForCausalLM.from_pretrained(expert_path)

tokenizer = tr.AutoTokenizer.from_pretrained(amateur_path)

user_message = """Give a very very brief docstring for the following function:\n```\nfunction updateEloScores(
	scores,
	results,
	kFactor = 4,
) {
	for (const result of results) {
		const { first, second, outcome } = result;
		const firstScore = scores[first] ?? 1000;
		const secondScore = scores[second] ?? 1000;

		const expectedScoreFirst = 1 / (1 + Math.pow(10, (secondScore - firstScore) / 400));
		const expectedScoreSecond = 1 / (1 + Math.pow(10, (firstScore - secondScore) / 400));
		let sa = 0.5;
		if (outcome === 1) {
			sa = 1;
		} else if (outcome === -1) {
			sa = 0;
		}
		scores[first] = firstScore + kFactor * (sa - expectedScoreFirst);
		scores[second] = secondScore + kFactor * (1 - sa - expectedScoreSecond);
	}
	return scores;
}\n```"""


prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_message},
    ],
    add_generation_prompt=True,
    tokenize=False,
)

def get_greedy_prob(expert_model, current_inputs):
    """Finds the next token with highest probability from the expert model. Returns this 
	probability and all the token probabilities."""
    with torch.no_grad():
		# get next token probs
        exp_outputs = expert_model(**current_inputs)
        exp_logits = exp_outputs.logits[:, -1, :]
        exp_probs = torch.softmax(exp_logits, dim=-1)
		# find greedy token
        greedy_token_id = torch.argmax(exp_probs, dim=-1).item()
        greedy_prob = exp_probs[0, greedy_token_id].item()

        if (testing):
	        greedy_token = tokenizer.decode([greedy_token_id])
	        print(f"Greedy token: '{greedy_token}' (ID: {greedy_token_id}), Probability: {greedy_prob:.4f}")

        return greedy_prob, exp_probs

def get_high_probability_tokens(probs, threshold, tokenizer):
    """Find high-probability tokens (p>=threshold). Returns a list of (token_id, probability) 
	tuples."""
    # make mask for tokens above threshold
    mask = probs[0] >= threshold
    
    # get masked probabilities
    token_ids = torch.where(mask)[0]
    probabilities = probs[0][mask]
    
    # return the high probability token IDs with their probabilities
    high_prob_tokens = []
    for token_id, prob in zip(token_ids, probabilities):
        high_prob_tokens.append((token_id.item(), prob.item()))

    if (testing):
        print(f"Tokens with prob >= {threshold:.4f} ({len(high_prob_tokens)} tokens):")
    
    return high_prob_tokens

def select_best_token_contrastive(high_prob_tokens, amateur_model, current_inputs, tokenizer):
    """Select the best token using batched evaluation."""
    if not high_prob_tokens:
        return None
    
    # extract token IDs for batch processing
    token_ids = [token[0] for token in high_prob_tokens]
    
    with torch.no_grad():
		# get amateur probabilities for all tokens
        amateur_outputs = amateur_model(**current_inputs)
        amateur_logits = amateur_outputs.logits[:, -1, :]
        all_amateur_probs = torch.softmax(amateur_logits, dim=-1)
        
        # get amateur probabilities for all high-probability tokens
        amateur_probs = all_amateur_probs[0, token_ids]
        expert_probs = torch.tensor([token[1] for token in high_prob_tokens])
        
        # compute objectives
        objectives = torch.log(expert_probs) - torch.log(amateur_probs)
        
        # find the best token
        best_idx = torch.argmax(objectives).item()
        best_token_id, best_prob = high_prob_tokens[best_idx]
        best_objective = objectives[best_idx].item()
        
        if (testing):
			# print all token evaluations with token text
            if hasattr(tokenizer, 'batch_decode'):
                token_texts = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
            else:
                token_texts = [tokenizer.decode([tid]) for tid in token_ids]
            for i, (token_id, prob) in enumerate(high_prob_tokens):
                print(f"  '{token_texts[i]}' (ID: {token_id}): exp_prob={prob:.4f}, ama_prob={amateur_probs[i].item():.4f}, objective={objectives[i].item():.4f}")
        
    if (testing):
		# print selected token
        print(f"\nSelected: '{tokenizer.decode([best_token_id])}' (ID: {best_token_id}) with prob={best_prob:.4f}, objective={best_objective:.4f}")

    return best_token_id

def contrastive_generation(amateur, expert, prompt, max_tokens, alpha=0.1) -> str:
    """Implements token-level contrastive decoding as described in Li et al."""
    # tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
    	# generate the contrastive response one token at a time
        generated_tokens = []
        current_inputs = inputs
        
        for _ in range(max_tokens):
            # step 1: Find the token with the highest probability (greedy_prob)
            greedy_prob, exp_probs = get_greedy_prob(expert, current_inputs)
            
            # step 2: Find all tokens with probability >= alpha * greedy_prob
            threshold = alpha * greedy_prob
            high_prob_tokens = get_high_probability_tokens(exp_probs, threshold, tokenizer)
            
            # step 3: Choose the high-probability token with the highest contrastive objective value
            best_token_id = select_best_token_contrastive(high_prob_tokens, amateur_model, current_inputs, tokenizer)

            # add the best token to the generated tokens
            next_token = torch.tensor([[best_token_id]])
            generated_tokens.append(next_token)
            # update inputs for next iteration
            new_input_ids = torch.cat([current_inputs["input_ids"], next_token], dim=1)
            new_attention_mask = torch.ones_like(new_input_ids)
            current_inputs = {
                "input_ids": new_input_ids,
                "attention_mask": new_attention_mask
            }
            
            # check for end token
            if best_token_id == tokenizer.eos_token_id:
                break
    
    # decode all generated tokens
    if generated_tokens:
        all_tokens = torch.cat(generated_tokens, dim=1)
        token_ids = all_tokens[0].tolist()
        full_response = tokenizer.decode(token_ids, skip_special_tokens=True)
    else:
        full_response = ""
    
    return full_response


print(contrastive_generation(amateur_model, expert_model, prompt, 50))
