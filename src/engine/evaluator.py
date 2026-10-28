from typing import Dict, List, Optional, Sequence

import torch
from tqdm import tqdm

from src.data.collator import batch_to_device, make_collator
from src.data.templates import IM_END
from src.eval.caption_metrics import caption_report
from src.eval.latency import LatencyTracker, count_generated_tokens
from src.eval.vqa_accuracy import clean_vqa_prediction, evaluate_vqa
from src.models.vlm import MiniVLM


class Evaluator:
    """
    Runs batched greedy generation over an eval dataset (left-padded, prompt-only items) and scores it.
    A dataset built with `use_image=False` gives the text-only baseline through the same code.
    """
    def __init__(self, vlm: MiniVLM, tokenizer, device, batch_size: int = 32):
        self.vlm, self.tok, self.device, self.batch_size = vlm, tokenizer, torch.device(device), batch_size
        self.eos_id = tokenizer.convert_tokens_to_ids(IM_END)

    @torch.no_grad()
    def generate_all(self, dataset, max_new_tokens: int, limit: Optional[int] = None, desc: str = "generating"):
        """Returns (texts, metas, latency summary), in dataset order."""
        n = min(limit or len(dataset), len(dataset))
        collate = make_collator(self.tok, "eval")
        tracker = LatencyTracker(self.device)
        texts: List[str] = []
        metas: List[dict] = []
        self.vlm.eval()
        for start in tqdm(range(0, n, self.batch_size), desc=desc, unit="batch"):
            items = [dataset[i] for i in range(start, min(start + self.batch_size, n))]
            batch = batch_to_device(collate(items), self.device)
            metas += batch.pop("meta")
            batch.pop("labels")
            with tracker.measure(len(items)) as span:
                generated = self.vlm.generate(**batch, max_new_tokens=max_new_tokens)
                span.tokens = count_generated_tokens(generated, self.eos_id, self.tok.pad_token_id)
            texts += self.tok.batch_decode(generated, skip_special_tokens=True)
        return texts, metas, tracker.summary()

    def run_vqa(self, dataset, max_new_tokens: int = 8, limit: Optional[int] = None) -> Dict[str, object]:
        texts, metas, latency = self.generate_all(dataset, max_new_tokens, limit, desc="vqa")
        result = evaluate_vqa(texts, metas)
        scores = result.pop("scores")
        rows = [
            {"question_id": m["question_id"], "image_id": m["image_id"], "question": m["question"],
             "prediction": t, "cleaned": clean_vqa_prediction(t), "answer": m["answer"],
             "answer_type": m["answer_type"], "question_type": m["question_type"], "score": s}
            for t, m, s in zip(texts, metas, scores)
        ]
        return {"metrics": result, "predictions": rows, "latency": latency}

    def run_captions(self, dataset, max_new_tokens: int = 40, limit: Optional[int] = None) -> Dict[str, object]:
        texts, metas, latency = self.generate_all(dataset, max_new_tokens, limit, desc="captions")
        refs = [m["references"] for m in metas]
        metrics = caption_report(texts, refs)
        rows = [{"image_id": m["image_id"], "prediction": t, "references": m["references"]}
                for t, m in zip(texts, metas)]
        return {"metrics": metrics, "predictions": rows, "latency": latency}


def constant_vqa_baseline(metas: Sequence[dict], answer: str) -> Dict[str, object]:
    """VQA accuracy of answering `answer` to every question (e.g. "yes"): a floor that needs no model."""
    result = evaluate_vqa([answer] * len(metas), list(metas))
    result.pop("scores")
    return result
