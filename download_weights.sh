mkdir -p weights

echo -n 'Do you want to use our trained model (y/n)? '
read -r answer
if [[ "${answer}" != "${answer#[Yy]}" ]];then
  echo -n "Downloading ShowHowTo weights ... "
  wget https://data.ciirc.cvut.cz/public/projects/2024ShowHowTo/weights/showhowto_2to8steps.pt -O weights/showhowto_2to8steps.pt --no-check-certificate
  SHA256SUM=$(sha256sum weights/showhowto_2to8steps.pt | cut -d' ' -f1)
  if [[ ${SHA256SUM} == "5759609fde82dc394a3e9872f145c50bed229d9d22d24dd682065e4e724ac47c" ]]; then
    echo "OK ✓"
  else
    echo "ERROR ✗"
    exit 1
  fi
fi

echo -n 'Do you want to train your own model (y/n)? '
read -r answer
if [[ "${answer}" != "${answer#[Yy]}" ]];then
  echo -n "Downloading DynamiCrafter weights ... "
  wget https://huggingface.co/Doubiiu/DynamiCrafter/resolve/main/model.ckpt -q -O ./weights/dynamicrafter_256_v1.ckpt
  SHA256SUM=$(sha256sum weights/dynamicrafter_256_v1.ckpt | cut -d' ' -f1)
  if [[ ${SHA256SUM} == "328d23963f1fe5af1324793117dfa80c8f5d3d31a2a7d5a6089a1c8aa72fb2da" ]]; then
    echo "OK ✓"
  else
    echo "ERROR ✗"
    exit 1
  fi
fi
