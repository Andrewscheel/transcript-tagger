import json
from pathlib import Path
session_total = 0
session_total_customer = 0
session_total_agent = 0

def load_transcript(file_path):
    with open(file_path, encoding="utf-8") as f:
        transcript = json.load(f)
    return transcript

def count_turns(transcript):
    customer_count = 0
    agent_count = 0
    for turn in transcript["turns"]:
        if turn["speaker"] == "customer":
            customer_count += 1
        elif turn["speaker"] == "agent":
            agent_count += 1
    return customer_count, agent_count

if __name__ == "__main__":
    paths = sorted(Path("data").glob("*.json"))
    for file in paths:

        transcript = load_transcript(file)
        session_customer_count, session_agent_count = count_turns(transcript)
    
        session_total += len(transcript['turns'])
        session_total_customer += session_customer_count
        session_total_agent += session_agent_count
        print(f"call_id: {transcript['call_id']}, total turns: {len(transcript['turns'])}, customer turns: {session_customer_count}, agent turns: {session_agent_count}")


    print(f"Total turns across all sessions: {session_total}")
    print(f"Customer turns across all sessions: {session_total_customer}")
    print(f"Agent turns across all sessions: {session_total_agent}")


