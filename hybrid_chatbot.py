#!/usr/bin/env python3
"""
hybrid_chatbot.py
Terminal-based hybrid (rule + ML) chatbot in pure Python.

Usage:
  1) Prepare an intents JSON (example below).
  2) Train:
       python hybrid_chatbot.py train intents.json
  3) Chat:
       python hybrid_chatbot.py chat
"""

import sys
import os
import json
import re
import math
import random
import argparse
from collections import defaultdict, Counter

MODEL_PATH = "hybrid_model.json"

# -----------------------
# Simple tokenizer / text utils
# -----------------------
def tokenize(text):
    # Lowercase, remove extra spaces, simple word split, keep numbers
    text = text.lower()
    # simple punctuation removal except for @/# if you want
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t]
    return tokens

def normalize_text(text):
    return " ".join(tokenize(text))

# -----------------------
# Rule-based subsystem
# -----------------------
class RuleEngine:
    def __init__(self, rules=None):
        # rules: list of dicts { "name":..., "pattern": regex or list of keywords, "response":..., "priority": int }
        self.rules = rules or []

    def match(self, text):
        text_low = text.lower()
        # exact phrase rules first (highest priority)
        # rules sorted by priority desc
        sorted_rules = sorted(self.rules, key=lambda r: r.get("priority", 0), reverse=True)
        for r in sorted_rules:
            patt = r.get("pattern")
            if isinstance(patt, str):
                # treat as substring search
                if patt.lower() in text_low:
                    return {"type": "rule", "rule_name": r.get("name"), "response": r.get("response"), "meta": r}
            elif isinstance(patt, list):
                # all keywords present?
                if all(kw.lower() in text_low for kw in patt):
                    return {"type": "rule", "rule_name": r.get("name"), "response": r.get("response"), "meta": r}
            elif hasattr(patt, "search"):
                if patt.search(text):
                    return {"type": "rule", "rule_name": r.get("name"), "response": r.get("response"), "meta": r}
        return None

# -----------------------
# Minimal Multinomial Naive Bayes (bag-of-words)
# -----------------------
class SimpleNB:
    def __init__(self):
        self.vocab = {}            # token -> index
        self.class_priors = {}     # label -> log prior
        self.token_log_probs = {}  # label -> list(log prob) length V
        self.labels = []

    def build_vocab(self, texts, min_freq=1):
        counter = Counter()
        for t in texts:
            counter.update(tokenize(t))
        # include tokens with freq >= min_freq
        tokens = [tok for tok, c in counter.items() if c >= min_freq]
        self.vocab = {tok: i for i, tok in enumerate(sorted(tokens))}
        return self.vocab

    def featurize(self, text):
        vec = [0] * len(self.vocab)
        for tok in tokenize(text):
            if tok in self.vocab:
                vec[self.vocab[tok]] += 1
        return vec

    def train(self, texts, labels, min_freq=1, alpha=1.0):
        # Build vocab
        self.build_vocab(texts, min_freq=min_freq)
        V = len(self.vocab)
        label_docs = defaultdict(list)
        for t, y in zip(texts, labels):
            label_docs[y].append(t)
        self.labels = sorted(label_docs.keys())
        N = len(texts)
        # class priors
        for y in self.labels:
            self.class_priors[y] = math.log(len(label_docs[y]) / N)
        # token counts per class
        for y in self.labels:
            counts = [0] * V
            total_count = 0
            for doc in label_docs[y]:
                vec = self.featurize(doc)
                for i, c in enumerate(vec):
                    counts[i] += c
                    total_count += c
            # apply Laplace smoothing and compute log probs
            denom = total_count + alpha * V
            log_probs = [(counts[i] + alpha) / denom for i in range(V)]
            # store log
            self.token_log_probs[y] = [math.log(p) for p in log_probs]

    def predict_proba(self, text):
        if not self.vocab:
            # no vocab -> uniform
            return {lbl: 1.0 / len(self.labels) for lbl in self.labels}
        vec = self.featurize(text)
        scores = {}
        for y in self.labels:
            score = self.class_priors.get(y, math.log(1e-9))
            probs = self.token_log_probs.get(y)
            if probs is None:
                scores[y] = score
                continue
            # add token contributions
            for i, count in enumerate(vec):
                if count > 0:
                    score += probs[i] * count
            scores[y] = score
        # convert log-scores to normalized probabilities
        max_score = max(scores.values())
        exps = {y: math.exp(scores[y] - max_score) for y in scores}
        s = sum(exps.values())
        probs = {y: exps[y] / s for y in exps}
        return probs

    def predict(self, text):
        probs = self.predict_proba(text)
        best = max(probs.items(), key=lambda kv: kv[1])
        return best  # (label, prob)

    def to_dict(self):
        return {
            "vocab": self.vocab,
            "class_priors": self.class_priors,
            "token_log_probs": self.token_log_probs,
            "labels": self.labels
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        obj.vocab = d["vocab"]
        obj.class_priors = d["class_priors"]
        obj.token_log_probs = d["token_log_probs"]
        obj.labels = d["labels"]
        return obj

# -----------------------
# Chatbot wrapper
# -----------------------
class HybridChatbot:
    def __init__(self, intents=None, rule_engine=None, nb_model=None, fallback_responses=None, min_confidence=0.6):
        # intents: dictionary loaded from JSON
        self.intents = intents or {}
        self.rule_engine = rule_engine or RuleEngine()
        self.nb = nb_model or SimpleNB()
        self.fallback_responses = fallback_responses or [
            "Sorry, I didn't get that. Can you rephrase?",
            "I'm not sure I understand — try asking differently.",
            "Hmm, I don't know that yet. You can teach me by adding an intent!"
        ]
        self.min_confidence = min_confidence

    def predict_intent(self, text):
        # 1) rule engine
        rule = self.rule_engine.match(text)
        if rule:
            return {"source": "rule", "intent": rule["rule_name"], "response": rule["response"], "score": 1.0}
        # 2) ML intent
        if not self.nb.labels:
            return {"source": "none", "intent": None, "response": None, "score": 0.0}
        label, prob = self.nb.predict(text)
        # if below threshold, return None
        if prob < self.min_confidence:
            return {"source": "ml", "intent": label, "response": None, "score": prob}
        # get a response template from intents
        intent_data = self.intents.get(label, {})
        resp = self.choose_response(intent_data.get("responses", []))
        return {"source": "ml", "intent": label, "response": resp, "score": prob}

    @staticmethod
    def choose_response(responses):
        if not responses:
            return None
        return random.choice(responses)

    def handle(self, text):
        # Preprocess basic cleaning
        text = text.strip()
        if not text:
            return "Say something — I'm listening."
        # rule/ML predict
        pred = self.predict_intent(text)
        if pred["response"]:
            return pred["response"] + f"  (via {pred['source']}, score={pred['score']:.2f})"
        else:
            # if ML predicted but low confidence or missing response, fallback
            if pred["source"] == "ml" and pred["intent"]:
                return f"I think you meant: '{pred['intent']}' (confidence {pred['score']:.2f}), but I'm not confident. " + random.choice(self.fallback_responses)
            return random.choice(self.fallback_responses)

# -----------------------
# Persistence: save/load model + intents + rules
# -----------------------
def save_model(path, intents, rules, nb_model):
    data = {
        "intents": intents,
        "rules": [],
        "nb": nb_model.to_dict()
    }
    # rules: cannot pickle compiled regex; so store as dict with pattern string and a flag
    for r in rules:
        patt = r.get("pattern")
        if hasattr(patt, "pattern"):
            patt_s = {"type": "regex", "source": patt.pattern}
        elif isinstance(patt, list):
            patt_s = {"type": "list", "source": patt}
        else:
            patt_s = {"type": "str", "source": patt}
        row = {
            "name": r.get("name"),
            "pattern": patt_s,
            "response": r.get("response"),
            "priority": r.get("priority", 0)
        }
        data["rules"].append(row)
    with open(path, "w", encoding="utf8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("Saved model to", path)

def load_model(path):
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)
    # rebuild rules
    rules = []
    for r in data.get("rules", []):
        patt = r["pattern"]
        if patt["type"] == "regex":
            pattern = re.compile(patt["source"], flags=re.IGNORECASE)
        elif patt["type"] == "list":
            pattern = patt["source"]
        else:
            pattern = patt["source"]
        rules.append({"name": r["name"], "pattern": pattern, "response": r["response"], "priority": r.get("priority", 0)})
    nb = SimpleNB.from_dict(data["nb"])
    intents = data.get("intents", {})
    return intents, rules, nb

# -----------------------
# Utility: build training data from intents JSON
# intents JSON format:
# {
#   "intent_name": {
#       "examples": ["sample sentence", ...],
#       "responses": ["...","..."]
#   },
#   ...
# }
# -----------------------
def build_training_set(intents):
    texts = []
    labels = []
    for intent_name, intent_data in intents.items():
        examples = intent_data.get("examples", [])
        for ex in examples:
            texts.append(normalize_text(ex))
            labels.append(intent_name)
    return texts, labels

# -----------------------
# Example rule definitions (you can expand)
# -----------------------
def default_rules():
    return [
        {"name": "greet_rule", "pattern": re.compile(r"\b(hi|hello|hey|good (morning|afternoon|evening))\b", re.IGNORECASE),
         "response": "Hello! How can I help you today?", "priority": 100},
        {"name": "bye_rule", "pattern": ["bye", "goodbye", "see you"], "response": "Goodbye! Have a nice day.", "priority": 100},
        {"name": "thanks", "pattern": ["thank", "thanks", "thx"], "response": "You're welcome!", "priority": 90},
        # phone number extraction as a rule example:
        {"name": "phone_number", "pattern": re.compile(r"\b\d{10}\b"), "response": "I detected a 10-digit number. Do you want me to save it?", "priority": 80},
    ]

# -----------------------
# CLI: train and chat
# -----------------------
def cmd_train(intents_path, model_path=MODEL_PATH):
    if not os.path.exists(intents_path):
        print("Intents file not found:", intents_path)
        return
    intents = json.load(open(intents_path, encoding="utf8"))
    texts, labels = build_training_set(intents)
    if not texts:
        print("No training examples found in intents file.")
        return
    nb = SimpleNB()
    nb.train(texts, labels, min_freq=1, alpha=1.0)
    rules = default_rules()
    save_model(model_path, intents, rules, nb)
    print("Training complete. Vocabulary size:", len(nb.vocab))
    # show a quick evaluation on training data
    correct = 0
    for t, y in zip(texts, labels):
        pred_label, prob = nb.predict(t)
        if pred_label == y:
            correct += 1
    print(f"Training accuracy (on training examples): {correct}/{len(labels)} = {correct/len(labels):.2f}")

def cmd_chat(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        print("Model not found. Run `python hybrid_chatbot.py train intents.json` first.")
        return
    intents, rules, nb = load_model(model_path)
    rule_engine = RuleEngine(rules)
    bot = HybridChatbot(intents=intents, rule_engine=rule_engine, nb_model=nb)
    print("Hybrid chatbot (type 'exit' or 'quit' to leave).")
    while True:
        try:
            text = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if text.lower() in ("exit", "quit"):
            print("Bye!")
            break
        reply = bot.handle(text)
        print("Bot:", reply)

# -----------------------
# main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "chat"], help="train or chat")
    parser.add_argument("intents", nargs="?", help="path to intents.json (for train)")
    parser.add_argument("--model", default=MODEL_PATH, help="path to save/load model")
    args = parser.parse_args()

    if args.mode == "train":
        if not args.intents:
            print("Please specify an intents JSON file: python hybrid_chatbot.py train intents.json")
            return
        cmd_train(args.intents, model_path=args.model)
    elif args.mode == "chat":
        cmd_chat(model_path=args.model)

if __name__ == "__main__":
    main()
