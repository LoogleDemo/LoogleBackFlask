#!/bin/bash
echo "export OPENAI_API_KEY=${OPENAI_API_KEY}" >> /etc/profile.d/my_env_vars.sh
echo "export GEM_API_KEY=${GEM_API_KEY}" >> /etc/profile.d/my_env_vars.sh
source /etc/profile.d/my_env_vars.sh
