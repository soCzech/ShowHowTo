mkdir -p weights

echo -n "Downloading ShowHowTo weights ... "
wget https://data.ciirc.cvut.cz/public/projects/2024ShowHowTo/weights/showhowto_2to8steps.pt -q -O weights/showhowto_2to8steps.pt
SHA256SUM=$(sha256sum weights/showhowto_2to8steps.pt | cut -d' ' -f1)
if [[ ${SHA256SUM} == "5759609fde82dc394a3e9872f145c50bed229d9d22d24dd682065e4e724ac47c" ]]; then
  echo "OK ✓"
else
  echo "ERROR ✗"
  exit 1
fi

