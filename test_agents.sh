#!/bin/bash

# AI Lego Bricks - Agent Testing Navigator
# Interactive script to explore and test different agent examples

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Project paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENTS_DIR="$PROJECT_ROOT/agent_orchestration/examples"
EXAMPLES_DIR="$PROJECT_ROOT/examples"
OUTPUT_DIR="$PROJECT_ROOT/output"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

print_header() {
    echo -e "${CYAN}================================"
    echo "🧱 AI Lego Bricks - Agent Tester"
    echo "================================${NC}"
    echo
}

print_agents() {
    echo -e "${BLUE}📋 Available Agent Examples:${NC}"
    echo
    
    local i=1
    for agent in "$AGENTS_DIR"/*.json; do
        if [[ -f "$agent" ]]; then
            local name=$(basename "$agent" .json)
            local desc=$(jq -r '.description // "No description"' "$agent" 2>/dev/null || echo "No description")
            echo -e "${GREEN}$i)${NC} ${YELLOW}$name${NC}"
            echo -e "   $desc"
            echo
            ((i++))
        fi
    done
}

print_runners() {
    echo -e "${BLUE}🚀 Available Runners:${NC}"
    echo
    echo -e "${GREEN}1)${NC} ${YELLOW}Simple Runner${NC} - Basic execution with results"
    echo -e "   Quick and clean agent execution"
    echo
    echo -e "${GREEN}2)${NC} ${YELLOW}Debug Runner${NC} - Verbose step-by-step debugging"
    echo -e "   Detailed execution info and troubleshooting"
    echo
    echo -e "${GREEN}3)${NC} ${YELLOW}Interactive Runner${NC} - Streaming, voice, conversation modes"
    echo -e "   Advanced interactions and real-time features"
    echo
}

select_agent() {
    local agents=("$AGENTS_DIR"/*.json)
    local agent_count=${#agents[@]}
    
    if [[ $agent_count -eq 0 ]]; then
        echo -e "${RED}❌ No agent files found in $AGENTS_DIR${NC}"
        exit 1
    fi
    
    while true; do
        read -p "Select agent (1-$agent_count): " choice
        
        if [[ "$choice" =~ ^[0-9]+$ ]] && [[ $choice -ge 1 ]] && [[ $choice -le $agent_count ]]; then
            local selected_agent="${agents[$((choice-1))]}"
            echo "$selected_agent"
            return
        else
            echo -e "${RED}Invalid choice. Please enter a number between 1 and $agent_count.${NC}"
        fi
    done
}

select_runner() {
    while true; do
        read -p "Select runner (1-3): " choice
        
        case $choice in
            1)
                echo "$EXAMPLES_DIR/run_agent_simple.py"
                return
                ;;
            2)
                echo "$EXAMPLES_DIR/run_agent_debug.py"
                return
                ;;
            3)
                echo "$EXAMPLES_DIR/run_agent_interactive.py"
                return
                ;;
            *)
                echo -e "${RED}Invalid choice. Please enter 1, 2, or 3.${NC}"
                ;;
        esac
    done
}

get_agent_inputs() {
    local agent_file="$1"
    local agent_name=$(basename "$agent_file" .json)
    
    echo -e "${BLUE}🔧 Configure inputs for $agent_name:${NC}"
    
    # Check what inputs this agent might need
    local needs_document=false
    local needs_voice=false
    local needs_user_input=false
    
    if grep -q "document_path\|pdf_path" "$agent_file"; then
        needs_document=true
    fi
    
    if grep -q "voice_input\|stt\|audio" "$agent_file"; then
        needs_voice=true
    fi
    
    if grep -q "user_input\|user_query\|input" "$agent_file"; then
        needs_user_input=true
    fi
    
    local inputs=()
    
    if [[ "$needs_document" == true ]]; then
        echo "This agent can process documents."
        read -p "Enter document path (or press Enter to skip): " doc_path
        if [[ -n "$doc_path" && -f "$doc_path" ]]; then
            inputs+=("--input" "$doc_path")
        fi
    fi
    
    if [[ "$needs_voice" == true ]]; then
        echo "This agent can process voice input."
        read -p "Enter voice file path (or press Enter to skip): " voice_path
        if [[ -n "$voice_path" && -f "$voice_path" ]]; then
            inputs+=("--input" "$voice_path")
        fi
    fi
    
    if [[ "$needs_user_input" == true ]]; then
        echo "This agent accepts text input."
        read -p "Enter your question/input (or press Enter to skip): " user_input
        if [[ -n "$user_input" ]]; then
            inputs+=("--user-input" "$user_input")
        fi
    fi
    
    echo "${inputs[@]}"
}

get_runner_options() {
    local runner="$1"
    local runner_name=$(basename "$runner" .py)
    local options=()
    
    case "$runner_name" in
        "run_agent_simple")
            read -p "Save output to file? (y/n): " save_output
            if [[ "$save_output" =~ ^[Yy] ]]; then
                options+=("--save-output")
            fi
            ;;
        "run_agent_debug")
            echo "Debug levels: 1=outputs only, 2=configs+inputs, 3=full details"
            read -p "Debug level (1-3, default 1): " debug_level
            debug_level=${debug_level:-1}
            options+=("--debug-level" "$debug_level")
            
            read -p "Save debug log? (y/n): " save_debug
            if [[ "$save_debug" =~ ^[Yy] ]]; then
                options+=("--save-debug")
            fi
            ;;
        "run_agent_interactive")
            echo "Interactive modes:"
            echo "1) Stream mode (for streaming agents)"
            echo "2) Voice mode (for voice agents)"
            echo "3) Conversation mode (continuous chat)"
            echo "4) File mode (file processing)"
            echo "5) Default interactive"
            
            read -p "Select mode (1-5, default 5): " mode
            case $mode in
                1) options+=("--stream") ;;
                2) options+=("--voice-mode") ;;
                3) options+=("--conversation") ;;
                4) options+=("--file-mode") ;;
            esac
            
            read -p "Save session? (y/n): " save_session
            if [[ "$save_session" =~ ^[Yy] ]]; then
                options+=("--save-session")
            fi
            ;;
    esac
    
    echo "${options[@]}"
}

run_agent() {
    local agent_file="$1"
    local runner="$2"
    local inputs=("${@:3}")
    
    local agent_name=$(basename "$agent_file" .json)
    local runner_name=$(basename "$runner" .py)
    
    echo -e "${PURPLE}🚀 Executing Agent${NC}"
    echo "Agent: $agent_name"
    echo "Runner: $runner_name"
    echo "Inputs: ${inputs[*]}"
    echo
    echo -e "${YELLOW}Press Ctrl+C to interrupt${NC}"
    echo "─────────────────────────────────"
    
    # Change to project directory
    cd "$PROJECT_ROOT"
    
    # Execute the agent
    python3 "$runner" "$agent_file" "${inputs[@]}"
    local exit_code=$?
    
    echo "─────────────────────────────────"
    if [[ $exit_code -eq 0 ]]; then
        echo -e "${GREEN}✅ Agent completed successfully!${NC}"
    else
        echo -e "${RED}❌ Agent failed with exit code $exit_code${NC}"
    fi
    
    return $exit_code
}

show_output_files() {
    echo -e "${BLUE}📁 Recent output files:${NC}"
    if [[ -d "$OUTPUT_DIR" ]]; then
        find "$OUTPUT_DIR" -type f -mtime -1 | head -10 | while read -r file; do
            local size=$(ls -lh "$file" | awk '{print $5}')
            local time=$(ls -l "$file" | awk '{print $6, $7, $8}')
            echo -e "${GREEN}📄${NC} $(basename "$file") ${YELLOW}($size)${NC} - $time"
        done
    else
        echo "No output files found."
    fi
}

main_menu() {
    while true; do
        print_header
        print_agents
        print_runners
        
        echo -e "${BLUE}🎯 Quick Actions:${NC}"
        echo -e "${GREEN}q)${NC} Quit"
        echo -e "${GREEN}o)${NC} Show recent output files"
        echo -e "${GREEN}c)${NC} Clear output directory"
        echo
        
        echo "Select an agent to test:"
        local selected_agent=$(select_agent)
        
        echo
        echo "Select a runner:"
        local selected_runner=$(select_runner)
        
        echo
        local inputs_array=($(get_agent_inputs "$selected_agent"))
        local options_array=($(get_runner_options "$selected_runner"))
        
        # Combine inputs and options
        local all_args=("${inputs_array[@]}" "${options_array[@]}")
        
        echo
        echo -e "${CYAN}Ready to execute!${NC}"
        read -p "Continue? (y/n): " confirm
        
        if [[ "$confirm" =~ ^[Yy] ]]; then
            run_agent "$selected_agent" "$selected_runner" "${all_args[@]}"
        fi
        
        echo
        read -p "Test another agent? (y/n): " again
        if [[ ! "$again" =~ ^[Yy] ]]; then
            break
        fi
        
        clear
    done
}

# Handle command line arguments for quick testing
if [[ $# -gt 0 ]]; then
    case "$1" in
        "list")
            print_header
            print_agents
            exit 0
            ;;
        "output")
            show_output_files
            exit 0
            ;;
        "clear")
            rm -rf "$OUTPUT_DIR"/*
            echo -e "${GREEN}✅ Output directory cleared${NC}"
            exit 0
            ;;
        *)
            echo "Usage: $0 [list|output|clear]"
            echo "  list   - Show available agents"
            echo "  output - Show recent output files"
            echo "  clear  - Clear output directory"
            echo "  (no args) - Interactive mode"
            exit 1
            ;;
    esac
fi

# Check dependencies
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ python3 is required but not installed${NC}"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo -e "${YELLOW}⚠️  jq not found - agent descriptions may not display properly${NC}"
fi

# Start interactive mode
main_menu
echo -e "${CYAN}👋 Thanks for testing AI Lego Bricks!${NC}"