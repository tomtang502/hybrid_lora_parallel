cd ~/
touch .git_token
sudo nano .git_token
# copy paste git token there, then ctrl +X then yes
chmod 600 ~/.git_token

GIT_TOKEN=$(<~/.git_token)

cd dlora