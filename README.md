# underactuated
underactuated robotic arm

Run docker
cd ./underactuated-arm
docker run -it \
  -v $(pwd):/home/underactuated-arm \
  -v $SSH_AUTH_SOCK:/ssh-agent \
  -e SSH_AUTH_SOCK=/ssh-agent \
  mujoco-cpp /bin/bash