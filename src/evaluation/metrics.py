"""Evaluation metrics for AFA extraction."""

import logging
from typing import Dict, List, Tuple, Set
import numpy as np
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score

logger = logging.getLogger(__name__)


def compute_metrics(
    prediction: Dict,
    gold_standard: Dict,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        prediction: Predicted AFA extraction
        gold_standard: Gold standard annotation
        
    Returns:
        Dict with all metrics
    """
    metrics = {}
    
    # Actor metrics
    if "actors" in prediction and "actors" in gold_standard:
        actor_metrics = compute_actor_metrics(
            prediction["actors"],
            gold_standard["actors"]
        )
        metrics.update({f"actor_{k}": v for k, v in actor_metrics.items()})
    
    # Frame metrics
    if "frames" in prediction and "frames" in gold_standard:
        frame_metrics = compute_frame_metrics(
            prediction["frames"],
            gold_standard["frames"]
        )
        metrics.update({f"frame_{k}": v for k, v in frame_metrics.items()})
    
    # Argument metrics
    if "arguments" in prediction and "arguments" in gold_standard:
        arg_metrics = compute_argument_metrics(
            prediction["arguments"],
            gold_standard["arguments"]
        )
        metrics.update({f"argument_{k}": v for k, v in arg_metrics.items()})
    
    return metrics


def compute_actor_metrics(
    predicted_actors: List[Dict],
    gold_actors: List[Dict],
) -> Dict[str, float]:
    """Compute actor identification and stance metrics."""
    # Extract actor names
    pred_names = set(a["name"].lower() for a in predicted_actors)
    gold_names = set(a["name"].lower() for a in gold_actors)
    
    # Pairwise F1 for actor identification
    if not gold_names:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "stance_acc": 0.0}
    
    tp = len(pred_names & gold_names)
    fp = len(pred_names - gold_names)
    fn = len(gold_names - pred_names)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Stance accuracy for matched actors
    matched_actors = pred_names & gold_names
    stance_correct = 0
    stance_total = 0
    
    pred_stance_map = {a["name"].lower(): a["stance"] for a in predicted_actors}
    gold_stance_map = {a["name"].lower(): a["stance"] for a in gold_actors}
    
    for actor_name in matched_actors:
        if pred_stance_map[actor_name] == gold_stance_map[actor_name]:
            stance_correct += 1
        stance_total += 1
    
    stance_acc = stance_correct / stance_total if stance_total > 0 else 0.0
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "stance_accuracy": float(stance_acc),
    }


def compute_frame_metrics(
    predicted_frames: Dict,
    gold_frames: Dict,
) -> Dict[str, float]:
    """Compute frame classification metrics."""
    metrics = {}
    
    # Primary frame accuracy
    pred_primary = predicted_frames.get("primary_frame")
    gold_primary = gold_frames.get("primary_frame")
    
    if pred_primary and gold_primary:
        metrics["primary_correct"] = float(pred_primary == gold_primary)
    else:
        metrics["primary_correct"] = 0.0
    
    # Secondary frame accuracy
    pred_secondary = predicted_frames.get("secondary_frame")
    gold_secondary = gold_frames.get("secondary_frame")
    
    if pred_secondary and gold_secondary:
        metrics["secondary_correct"] = float(pred_secondary == gold_secondary)
    elif not pred_secondary and not gold_secondary:
        metrics["secondary_correct"] = 1.0  # Both correctly null
    else:
        metrics["secondary_correct"] = 0.0
    
    return metrics


def compute_argument_metrics(
    predicted_args: Dict,
    gold_args: Dict,
) -> Dict[str, float]:
    """Compute argument extraction metrics (span-level)."""
    metrics = {}
    
    # Claim F1
    if "claim" in predicted_args and "claim" in gold_args:
        claim_f1 = span_f1(
            predicted_args["claim"],
            gold_args["claim"]
        )
        metrics["claim_f1"] = claim_f1
    
    # Evidence F1 (list of evidence pieces)
    if "evidence" in predicted_args and "evidence" in gold_args:
        pred_evidence = " ".join(predicted_args["evidence"])
        gold_evidence = " ".join(gold_args["evidence"])
        evidence_f1 = span_f1(pred_evidence, gold_evidence)
        metrics["evidence_f1"] = evidence_f1
    
    # Warrant F1
    if "warrant" in predicted_args and "warrant" in gold_args:
        warrant_f1 = span_f1(
            predicted_args["warrant"],
            gold_args["warrant"]
        )
        metrics["warrant_f1"] = warrant_f1
    
    return metrics


def span_f1(predicted_text: str, gold_text: str) -> float:
    """
    Compute token-level F1 for span matching.
    
    Args:
        predicted_text: Predicted span
        gold_text: Gold span
        
    Returns:
        F1 score
    """
    if not predicted_text or not gold_text:
        return 0.0
    
    # Tokenize
    pred_tokens = set(predicted_text.lower().split())
    gold_tokens = set(gold_text.lower().split())
    
    if not gold_tokens:
        return 0.0
    
    # Compute F1
    tp = len(pred_tokens & gold_tokens)
    fp = len(pred_tokens - gold_tokens)
    fn = len(gold_tokens - pred_tokens)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return float(f1)


def krippendorff_alpha(
    annotations: List[List[int]],
    level: str = "nominal",
) -> float:
    """
    Compute Krippendorff's alpha for inter-annotator agreement.
    
    Args:
        annotations: List of annotator judgments (list of lists)
        level: Measurement level ('nominal', 'ordinal', 'interval')
        
    Returns:
        Alpha coefficient
    """
    # Simplified implementation
    # For production, use krippendorff library
    
    import krippendorff
    
    # Convert to numpy array
    data = np.array(annotations)
    
    alpha = krippendorff.alpha(
        reliability_data=data,
        level_of_measurement=level,
    )
    
    return float(alpha)


def compute_iaa(
    annotations: List[Dict],
    component: str,
) -> Dict[str, float]:
    """
    Compute inter-annotator agreement metrics.
    
    Args:
        annotations: List of annotations from different annotators
        component: Component type ('actor', 'stance', 'frame', 'argument')
        
    Returns:
        Dict with IAA metrics
    """
    if component == "stance":
        # Krippendorff's alpha for nominal data
        stance_labels = []
        for ann in annotations:
            labels = [a["stance"] for a in ann.get("actors", [])]
            stance_labels.append(labels)
        
        # Convert to numeric
        stance_map = {"supportive": 0, "opposing": 1, "neutral": 2, "mixed": 3}
        numeric_labels = []
        for labels in stance_labels:
            numeric = [stance_map.get(l, 2) for l in labels]
            numeric_labels.append(numeric)
        
        alpha = krippendorff_alpha(numeric_labels, level="nominal")
        
        return {"krippendorff_alpha": float(alpha)}
    
    elif component == "frame":
        # Krippendorff's alpha for frames
        frame_labels = []
        for ann in annotations:
            frame = ann.get("frames", {}).get("primary_frame")
            frame_labels.append(frame)
        
        # Convert to numeric
        frame_map = {
            "economic_opportunity": 0,
            "economic_risk": 1,
            "regulatory_compliance": 2,
            "technological_solution": 3,
            "environmental_urgency": 4,
            "social_responsibility": 5,
            "market_dynamics": 6,
            "uncertainty_skepticism": 7,
        }
        
        numeric = [[frame_map.get(f, 0)] for f in frame_labels]
        alpha = krippendorff_alpha(numeric, level="nominal")
        
        return {"krippendorff_alpha": float(alpha)}
    
    elif component == "actor":
        # Pairwise F1 for actor spans
        pairwise_f1s = []
        
        for i in range(len(annotations)):
            for j in range(i + 1, len(annotations)):
                actors_i = set(a["name"].lower() for a in annotations[i].get("actors", []))
                actors_j = set(a["name"].lower() for a in annotations[j].get("actors", []))
                
                if not actors_i and not actors_j:
                    continue
                
                tp = len(actors_i & actors_j)
                fp = len(actors_i - actors_j)
                fn = len(actors_j - actors_i)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                pairwise_f1s.append(f1)
        
        avg_f1 = np.mean(pairwise_f1s) if pairwise_f1s else 0.0
        
        return {"pairwise_f1": float(avg_f1)}
    
    return {}
