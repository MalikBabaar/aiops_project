import re
import json
import pandas as pd

# Replace or insert this function:
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

parsed_logs = []

# Read from Filebeat output
with open('/var/log/filebeat/jboss_parsed.json', 'r') as f:
    for line in f:
        try:
            log = json.loads(line)
            parsed = parse_line(log.get('message', ''))
            if parsed:
                parsed_logs.append(parsed)
        except Exception as e:
            print(f"Error: {e}")

df = pd.DataFrame(parsed_logs)
df.to_csv("jboss_parsed_logs.csv", index=False)
print("Parsed logs saved to jboss_parsed_logs.csv")
