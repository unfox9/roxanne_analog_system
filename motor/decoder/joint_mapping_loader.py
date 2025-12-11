import re

def load_action_channels(path):
    channels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            idx = int(parts[0])
            name = parts[1]
            axis = parts[2].upper()
            channels.append({"index": idx, "name": name, "axis": axis})
    channels.sort(key=lambda c: c["index"])
    return channels
