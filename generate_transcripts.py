import csv
import json
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()

MODEL = "claude-haiku-5-5"
DATA_DIR = Path("data")  # change this if stats.py reads a different folder
LABELS_FILE = DATA_DIR / "labels.csv"
COMPANY = "a generic telecom and internet provider (do not name a real company)"

# Set to a small number like 3 for a test run, or None to do all 20.
LIMIT = None
MAX_TOKENS = 8000  # max for claude-haiku-5-5 is 8192, but we leave some room for the prompt

# (intent, outcome, channel, difficulty, style, twist)
PLAN = [
    ("order_status", "resolved", "voice", "clean", "ivr", "no_match"),
    ("order_status", "resolved", "chat", "clean", "ivr", ""),
    ("billing_dispute", "escalated", "voice", "clean", "ivr", "asks_agent"),
    ("billing_dispute", "resolved", "voice", "clean", "llm", ""),
    ("account_access", "resolved", "voice", "clean", "ivr", "no_match"),
    ("account_access", "escalated", "chat", "clean", "ivr", "asks_agent"),
    ("service_outage", "resolved", "voice", "clean", "ivr", "troubleshoot"),
    ("service_outage", "queue_transfer", "voice", "clean", "ivr", "troubleshoot"),
    ("cancel_service", "escalated", "voice", "clean", "ivr", "retention_decline"),
    ("cancel_service", "exited_bot", "voice", "clean", "ivr", "no_match"),
    ("plan_change", "resolved", "chat", "clean", "ivr", ""),
    ("plan_change", "queue_transfer", "voice", "clean", "llm", ""),
    ("product_question", "resolved", "chat", "clean", "llm", ""),
    ("product_question", "exited_bot", "chat", "clean", "ivr", "no_match"),
    ("billing_dispute+plan_change", "queue_transfer", "voice", "multi_intent", "ivr", ""),
    ("order_status+billing_dispute", "escalated", "voice", "multi_intent", "ivr", "asks_agent"),
    ("account_access", "resolved", "chat", "ambiguous", "ivr", ""),
    ("cancel_service", "exited_queue", "voice", "ambiguous", "llm", ""),
    ("none", "exited_bot", "voice", "no_intent", "ivr", "silent"),
    ("none", "escalated", "voice", "no_intent", "ivr", "only_agent"),
]

STYLE_RULES = {
    "ivr": """Scripted intent-based virtual agent (IVA), not a free-form AI:
- Agent turns are one or two short sentences. It asks for one thing at a time.
- Agent can only: verify an account id, look up order status, check for an outage, walk through basic troubleshooting, answer a few canned product questions, or transfer. It never invents other capabilities.
- Agent may greet the customer by a fake first name (for example "Hi Andrew, how can I help you today?").
- Two agent turns in a row are allowed (for example "Sorry, I didn't get that." then the re-prompt as a separate turn).
- Agent uses short acknowledgements like "OK, I can help with that." and "Thank you."
Style examples (shape only, do not copy):
  agent: please tell me your account id
  customer: hold on
  agent: Sorry, I didn't get that.
  agent: please tell me your account id
  customer: 00024527
  agent: thank you

  agent: Hi Andrew, how can I help you today?
  customer: id like to cancel my account
  agent: OK, I can help with that.
  agent: Can you help me understand why you'd like to cancel?
  customer: no
  agent: OK, let me put you through to one of my colleagues.""",
    "llm": """Modern LLM-backed assistant: fluent, natural, longer agent sentences, handles free-form requests smoothly. Still no invented account data beyond what a customer would plausibly be told.""",
}

CHANNEL_RULES = {
    "voice": "Voice call. Customer turns look like speech recognition output: terse, lowercase, mostly no punctuation, spoken numbers written as digit strings, occasional filler or mis-hearing.",
    "chat": "Web chat. Customer turns are short, lowercase, with occasional typos and abbreviations.",
}

DIFFICULTY_RULES = {
    "clean": "The customer states what they want clearly in their first turn.",
    "multi_intent": "The customer's first turn contains two requests. The first intent listed is the main one.",
    "ambiguous": "The customer's first turn is vague and could fit several intents (for example 'it's not working'). The true intent only becomes clear after a clarifying question.",
    "no_intent": "The customer never states a clear request. They are off-topic, unclear or silent. The agent tries to clarify and does not succeed in identifying an intent.",
}

OUTCOME_RULES = {
    "resolved": "The issue is handled and the customer confirms or thanks the agent. The conversation ends.",
    "escalated": "The virtual agent tells the customer it is transferring them to a live human and the transcript ends there. No human turns are included.",
    "exited_bot": "The customer gives up or hangs up during the virtual agent interaction, with no resolution and no handoff. Keep it short (3 to 6 turns).",
    "queue_transfer": "The customer is told they are being moved to another department's queue, and the conversation ends there.",
    "exited_queue": "The virtual agent places the customer in a queue for a live agent. The customer says they are giving up or leaving, and the transcript ends there. No human turns are included.",
}

TWIST_RULES = {
    "": "",
    "no_match": "At least once the customer says something the agent cannot use (for example 'hold on' or an unrelated remark). The agent replies 'Sorry, I didn't get that.' (or similar) as one turn, then repeats its prompt as a separate turn.",
    "asks_agent": "At some point the customer's text on a turn is exactly the single word: agent. This is how they ask for a human. Handle it according to the outcome.",
    "only_agent": "The customer's only turn in the whole transcript has the text exactly: agent. The agent greets, the customer says 'agent', and the agent responds according to the outcome. There are no other customer turns.",
    "silent": "The customer gives no input for at least two turns. Write each of those customer turns with the exact text [no_input]. The agent re-prompts after each one.",
    "troubleshoot": "The agent walks the customer through two common fixes one at a time (for example restart the modem, then check the cables), asking after each whether it worked. Handle the end according to the outcome.",
    "retention_decline": "The agent asks why the customer wants to cancel. The customer declines to say (for example 'no'), and the flow continues to the outcome.",
}


def build_prompt(intent, outcome, channel, difficulty, style, twist):
    twist_line = f"- Special requirement: {TWIST_RULES[twist]}\n" if twist else ""
    return f"""Write a realistic synthetic customer service conversation as JSON.

Setup:
- Company: {COMPANY}
- Customer intent: {intent}
- Channel: {CHANNEL_RULES[channel]}
- Difficulty: {DIFFICULTY_RULES[difficulty]}
- Outcome: {OUTCOME_RULES[outcome]}
{twist_line}
Agent style:
{STYLE_RULES[style]}

Rules:
- Speakers are only "agent" and "customer". The agent is an automated virtual agent.
- The first turn is the agent greeting.
- 4 to 10 turns in total, unless a rule above says shorter.
- Use fake details only (for example order 48213). No real names of companies, phone numbers or addresses.
- Do not mention the intent, outcome or these instructions in the text.

Return only a JSON array, with no other text. Each item looks like:
{{"speaker": "agent", "text": "..."}}
"""


def generate_one(row):
    intent, outcome, channel, difficulty, style, twist = row
    prompt = build_prompt(intent, outcome, channel, difficulty, style, twist)
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason == "max_tokens":
        raise ValueError("reply was cut off by max_tokens")

    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    turns = json.loads(text)
    return turns, response.usage.input_tokens, response.usage.output_tokens


def is_valid(turns):
    if not isinstance(turns, list) or not (3 <= len(turns) <= 14):
        return False
    for turn in turns:
        if not isinstance(turn, dict):
            return False
        if turn.get("speaker") not in ("agent", "customer"):
            return False
        if not isinstance(turn.get("text"), str) or not turn["text"].strip():
            return False
    return True


def main():
    DATA_DIR.mkdir(exist_ok=True)
    total_in = 0
    total_out = 0

    for i, row in enumerate(PLAN[:LIMIT], start=1):
        call_id = f"T{100 + i}"
        path = DATA_DIR / f"{call_id}.json"

        if path.exists():
            print(f"{call_id}: already exists, skipping")
            continue

        try:
            turns, tokens_in, tokens_out = generate_one(row)
        except Exception as error:  # broad on purpose for now; we tighten this in week 3
            print(f"{call_id}: failed ({error})")
            continue

        if not is_valid(turns):
            print(f"{call_id}: output did not match the schema, skipped")
            continue

        channel = row[2]
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"call_id": call_id, "channel": channel, "turns": turns}, f, indent=2)

        total_in += tokens_in
        total_out += tokens_out
        print(f"{call_id}: saved ({len(turns)} turns)")

    # Answer key: what the tagger should find. Channel is metadata, so it is not here.
    with open(LABELS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["call_id", "intent", "outcome", "difficulty", "style", "twist"])
        for i, row in enumerate(PLAN, start=1):
            call_id = f"T{100 + i}"
            if (DATA_DIR / f"{call_id}.json").exists():
                intent, outcome, _channel, difficulty, style, twist = row
                writer.writerow([call_id, intent, outcome, difficulty, style, twist])

    print(f"Tokens used this run: {total_in} in, {total_out} out")


if __name__ == "__main__":
    main()