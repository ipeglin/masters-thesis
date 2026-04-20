#!/bin/bash
# sys-logger.sh - Centralized logging with colors and structure

if [ -t 1 ]; then
    C_RESET='\033[0m'
    C_BOLD='\033[1m'
    C_BLUE='\033[0;34m'
    C_GREEN='\033[0;32m'
    C_YELLOW='\033[0;33m'
    C_RED='\033[0;31m'
    C_CYAN='\033[0;36m'
else
    C_RESET='' C_BOLD='' C_BLUE='' C_GREEN='' C_YELLOW='' C_RED='' C_CYAN=''
fi

log_title()   { echo -e "\n${C_BOLD}${C_BLUE}=== ${1} ===${C_RESET}"; }
log_step()    { echo -e "${C_BOLD}${C_CYAN}➜${C_RESET} ${C_BOLD}${1}${C_RESET}"; }
log_info()    { echo -e "  ${C_CYAN}ℹ${C_RESET} ${1}"; }
log_success() { echo -e "  ${C_GREEN}✔${C_RESET} ${1}"; }
log_warn()    { echo -e "  ${C_YELLOW}⚠ WARNING: ${1}${C_RESET}"; }
log_err()     { echo -e "  ${C_RED}✖ ERROR: ${1}${C_RESET}"; }
