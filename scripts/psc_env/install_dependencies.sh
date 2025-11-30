# MAKE SURE TO RUN
# module load anaconda3
# conda activate dlora

# 1) One-time per shell/sbatch: create an askpass helper
ASKPASS="$(mktemp)"; cat > "$ASKPASS" <<'EOF'
#!/bin/sh
case "$1" in
  Username*) printf '%s\n' "${GITHUB_USER:-x-oauth-basic}" ;;
  Password*) printf '%s\n' "${GIT_TOKEN:?missing}" ;;
esac
EOF
chmod 700 "$ASKPASS"

# 2) Export your token (stored securely in ~/.git_token with chmod 600)
export GIT_TOKEN="$(tr -d '\n' < "$HOME/.git_token")"
export GIT_ASKPASS="$ASKPASS"
export GIT_TERMINAL_PROMPT=0

# 3) Now pip can install from private git URLs in requirements.txt
pip install -e .

# 4) Optional cleanup when done
rm -f "$ASKPASS"; unset GIT_TOKEN GIT_ASKPASS