#!/usr/bin/env python3
"""
Log viewer utility for agent logs
"""
import json
import argparse
import os
from datetime import datetime


def format_log_entry(entry):
    """Format a log entry for readable display"""
    timestamp = entry.get("timestamp", "Unknown")
    stage = entry.get("stage", "Unknown")
    message = entry.get("message", "No message")
    data = entry.get("data", {})

    # Format timestamp
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        formatted_time = dt.strftime("%H:%M:%S")
    except Exception:
        formatted_time = timestamp

    print(f"\n[{formatted_time}] {stage}")
    print(f"  {message}")

    # Show key data fields
    if data:
        for key, value in data.items():
            if key == "response" and isinstance(value, str) and len(value) > 200:
                # Show model responses with proper formatting
                print(f"  📝 {key}:")
                print("    " + "\n    ".join(value.split("\n")))
            elif key == "prompt_preview" and isinstance(value, str):
                print(f"  💬 {key}: {value[:100]}{'...' if len(value) > 100 else ''}")
            elif key in ["headers", "model", "processing_time", "status"]:
                print(f"  📊 {key}: {value}")


def view_log_file(log_file, filter_stage=None, show_responses=True):
    """View a log file with optional filtering"""
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        return

    print(f"📋 Viewing log file: {log_file}")
    print("=" * 60)

    try:
        with open(log_file, "r") as f:
            entries = []
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue

        filtered_entries = entries
        if filter_stage:
            filtered_entries = [
                e for e in entries if filter_stage.lower() in e.get("stage", "").lower()
            ]

        if not show_responses:
            # Remove model responses for cleaner view
            for entry in filtered_entries:
                if "data" in entry and "response" in entry["data"]:
                    entry["data"] = {
                        k: v for k, v in entry["data"].items() if k != "response"
                    }

        for entry in filtered_entries:
            format_log_entry(entry)

        print(f"\n📊 Summary: {len(filtered_entries)} entries displayed")
        if filter_stage:
            print(f"🔍 Filter: {filter_stage}")

    except Exception as e:
        print(f"❌ Error reading log file: {e}")


def list_log_files():
    """List available log files"""
    log_dir = "agent_logs"
    if not os.path.exists(log_dir):
        print(f"❌ Log directory not found: {log_dir}")
        return

    log_files = [f for f in os.listdir(log_dir) if f.endswith(".log")]
    if not log_files:
        print(f"📁 No log files found in {log_dir}")
        return

    print(f"📁 Available log files in {log_dir}:")
    for i, log_file in enumerate(sorted(log_files), 1):
        full_path = os.path.join(log_dir, log_file)
        size = os.path.getsize(full_path)
        mtime = datetime.fromtimestamp(os.path.getmtime(full_path))
        print(f"  {i}. {log_file} ({size} bytes, {mtime.strftime('%Y-%m-%d %H:%M')})")


def main():
    parser = argparse.ArgumentParser(description="View agent log files")
    parser.add_argument("log_file", nargs="?", help="Log file to view")
    parser.add_argument(
        "--list", "-l", action="store_true", help="List available log files"
    )
    parser.add_argument("--filter", "-f", help="Filter by stage name")
    parser.add_argument(
        "--no-responses", "-n", action="store_true", help="Hide model responses"
    )

    args = parser.parse_args()

    if args.list:
        list_log_files()
        return

    if not args.log_file:
        print("Please specify a log file or use --list to see available files")
        list_log_files()
        return

    # If just filename provided, look in agent_logs directory
    if not os.path.dirname(args.log_file):
        args.log_file = os.path.join("agent_logs", args.log_file)

    view_log_file(args.log_file, args.filter, not args.no_responses)


if __name__ == "__main__":
    main()
