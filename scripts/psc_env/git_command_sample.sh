# prepare once per shell/sbatch
ASKPASS="$(mktemp)"; cat > "$ASKPASS" <<'EOF'
#!/bin/sh
case "$1" in
  Username*) printf '%s\n' "${GITHUB_USER:-x-oauth-basic}" ;;
  Password*) printf '%s\n' "${GIT_TOKEN:?missing}" ;;
esac
EOF
chmod 700 "$ASKPASS"

export GIT_TOKEN="$(tr -d '\n' < "$HOME/.git_token")"
export GIT_ASKPASS="$ASKPASS"
export GIT_TERMINAL_PROMPT=0

# GIT COMMAND HERE
git pull         

# GIT COMMAND END

# (optional) cleanup when done:
rm -f "$ASKPASS"; unset GIT_TOKEN GIT_ASKPASS