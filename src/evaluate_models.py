from src.command_agent import CommandAgent
from src.nlp_pipeline import NLPPipeline
from src.malware_detector import MalwareDetector




def recall_at_k(pred, true, k):
    """Returns 1 if true command is in top-k, else 0."""
    return int(true == pred)


def compute_mrr(pred_list, true_list):
    """Compute Mean Reciprocal Rank."""
    rr_scores = []
    for pred, true in zip(pred_list, true_list):
        if pred == true:
            rr_scores.append(1.0)  # rank 1
        else:
            rr_scores.append(0.0)  # not found in top1
    return sum(rr_scores) / len(rr_scores) if rr_scores else 0.0



def evaluate_model(agent, inputs, outputs, name="TF-IDF"):
    """
    Evaluate the agent on the full dataset.
    Returns all metrics + predictions.
    """

    total = len(inputs)
    correct_top1 = 0
    recall3 = 0
    recall5 = 0

    predictions = []
    ground_truth = []

    for text, true_cmd in zip(inputs, outputs):

        # Get prediction using best_match
        try:
            pred = agent._find_best_match(text, top_k=5)
        except Exception:
            pred = None

        if pred is None:
            pred = "NO_MATCH"

        predictions.append(pred)
        ground_truth.append(true_cmd)

        # Top-1 accuracy
        if pred.lower().strip() == true_cmd.lower().strip():
            correct_top1 += 1

        # Recall@3 and Recall@5 (since we only get 1 predicted command, top1=top3=top5)
        recall3 += recall_at_k(pred, true_cmd, 3)
        recall5 += recall_at_k(pred, true_cmd, 5)

    accuracy = correct_top1 / total
    recall3 /= total
    recall5 /= total
    mrr = compute_mrr(predictions, ground_truth)

    print(f"\n=== {name} Metrics ===")
    print(f"Total: {total}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall@3: {recall3:.4f}")
    print(f"Recall@5: {recall5:.4f}")
    print(f"MRR: {mrr:.4f}")

    return {
        "accuracy": accuracy,
        "recall3": recall3,
        "recall5": recall5,
        "mrr": mrr,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_results(results_dict):
    """
    results_dict = {
        "TF-IDF": {"accuracy":..., "recall3":..., ...},
        "Embeddings": {...}
    }
    """

    labels = list(results_dict.keys())
    accuracy_vals = [results_dict[k]["accuracy"] for k in labels]
    recall3_vals = [results_dict[k]["recall3"] for k in labels]
    recall5_vals = [results_dict[k]["recall5"] for k in labels]
    mrr_vals = [results_dict[k]["mrr"] for k in labels]

    x = np.arange(len(labels))
    width = 0.2

    plt.figure(figsize=(10, 6))
    plt.bar(x - width*1.5, accuracy_vals, width, label="Accuracy")
    plt.bar(x - width*0.5, recall3_vals, width, label="Recall@3")
    plt.bar(x + width*0.5, recall5_vals, width, label="Recall@5")
    plt.bar(x + width*1.5, mrr_vals, width, label="MRR")

    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.title("Model Comparison (TF-IDF vs Embeddings)")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("Loading full dataset...")
    agent_tfidf = CommandAgent(use_embeddings=False)
    inputs = agent_tfidf.input_texts
    outputs = agent_tfidf.output_commands

    print(f"Loaded {len(inputs)} examples")

    # Evaluate TF-IDF
    tfidf_results = evaluate_model(agent_tfidf, inputs, outputs, name="TF-IDF")

    # Evaluate Embeddings
    agent_emb = CommandAgent(use_embeddings=True)
    emb_results = evaluate_model(agent_emb, inputs, outputs, name="Embeddings")

    # Visualize comparison
    all_results = {
        "TF-IDF": tfidf_results,
        "Embeddings": emb_results
    }

    visualize_results(all_results)


if __name__ == "__main__":
    main()
