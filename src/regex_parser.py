import re
import json
import pandas as pd

def parse_line(line):
    pattern = re.compile(
        r'^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+[\+\-]\d{2}:\d{2})\s+'
        r'(?P<host>[^\s]+)\s+'
        r'(?P<component>[^\s]+)\[(?P<pid>\d+)\]:\s+'
        r'(?P<message>.*)$'
    )
    match = pattern.match(line)
    if match:
        return match.groupdict()
    return None


# 🔹 Change this to your dataset filename
input_file = "jboss_parsed.json"   # make sure this file exists in the same folder
output_file = "jboss_parsed_logs.csv"

parsed_logs = []

with open(input_file, 'r') as f:
    for line in f:
        try:
            log = json.loads(line)   # JSON object per line
            parsed = parse_line(log.get('message', ''))
            if parsed:
                parsed_logs.append(parsed)
        except Exception as e:
            print(f"Error: {e}")

# Save to CSV
df = pd.DataFrame(parsed_logs)
df.to_csv(output_file, index=False)

print(f"✅ Parsed logs saved to {output_file}")
