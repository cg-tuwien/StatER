#!/usr/bin/env bash

# © 2025 Hiroyuki Sakai

dpkg -s unzip &> /dev/null

if [ $? -ne 0 ]; then
  echo "This script requires unzip, which was not found on your system."
  read -p "Do you want to install it now? You may have to provide your password. (y/n)" unzip_choice
  if [[ "$unzip_choice" =~ ^[Yy]$ ]]; then
    sudo apt update
    sudo apt install unzip
  else
    echo "Aborting."
    exit 0
  fi
fi

cd scenes/

declare -A urls
urls["veach-bidir"]="https://benedikt-bitterli.me/resources/pbrt-v3/veach-bidir.zip"
urls["furball"]="https://benedikt-bitterli.me/resources/pbrt-v3/furball.zip"
urls["car"]="https://benedikt-bitterli.me/resources/pbrt-v3/car.zip"
urls["bathroom"]="https://benedikt-bitterli.me/resources/pbrt-v3/bathroom.zip"
urls["straight-hair"]="https://benedikt-bitterli.me/resources/pbrt-v3/straight-hair.zip"
urls["curly-hair"]="https://benedikt-bitterli.me/resources/pbrt-v3/curly-hair.zip"
urls["lamp"]="https://benedikt-bitterli.me/resources/pbrt-v3/lamp.zip"
urls["glass-of-water"]="https://benedikt-bitterli.me/resources/pbrt-v3/glass-of-water.zip"
urls["house"]="https://benedikt-bitterli.me/resources/pbrt-v3/house.zip"
urls["classroom"]="https://benedikt-bitterli.me/resources/pbrt-v3/classroom.zip"
urls["kitchen"]="https://benedikt-bitterli.me/resources/pbrt-v3/kitchen.zip"
urls["measure-one"]="https://owncloud.tuwien.ac.at/index.php/s/2rJf3TKUcnMBiAF/download"
urls["spaceship"]="https://benedikt-bitterli.me/resources/pbrt-v3/spaceship.zip"

for url in "${!urls[@]}"
do
  wget -O "${url}.zip" "${urls[${url}]}"
  unzip -o "${url}.zip"
  find "${url}" -type d -exec chmod 775 {} +
  rm "${url}.zip"
done

cd ../
