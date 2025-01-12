echo "Downloading ShowHowTo dataset to './data'."

echo -n "Downloading ShowHowTo test set ... "
mkdir -p data/ShowHowToTest
cd data/ShowHowToTest || exit 1
wget https://data.ciirc.cvut.cz/public/projects/2024ShowHowTo/dataset/ShowHowToTest.prompts.tar.gz --no-check-certificate -q
wget https://data.ciirc.cvut.cz/public/projects/2024ShowHowTo/dataset/ShowHowToTest.keyframes.tar.gz --no-check-certificate -q

tar xf ShowHowToTest.prompts.tar.gz && rm ShowHowToTest.prompts.tar.gz && \
  tar xf ShowHowToTest.keyframes.tar.gz && rm ShowHowToTest.keyframes.tar.gz && \
  echo "OK ✓" || echo "ERROR ✗"

cd ../..

echo -n "Downloading ShowHowTo train set ... "
mkdir -p data/ShowHowToTrain
cd data/ShowHowToTrain || exit 1
wget https://data.ciirc.cvut.cz/public/projects/2024ShowHowTo/dataset/ShowHowToTrain.prompts.tar.gz --no-check-certificate -q
wget https://data.ciirc.cvut.cz/public/projects/2024ShowHowTo/dataset/ShowHowToTrain.keyframes.tar.gz --no-check-certificate -q

tar xf ShowHowToTrain.prompts.tar.gz && rm ShowHowToTrain.prompts.tar.gz && \
  tar xf ShowHowToTrain.keyframes.tar.gz && rm ShowHowToTrain.keyframes.tar.gz && \
  echo "OK ✓" || echo "ERROR ✗"

cd ../..
cd data

echo -n 'Do you want to also download image sequences? Approximately 200GB will be downloaded, password is required. (y/n) '
read -r answer
if [[ "${answer}" != "${answer#[Yy]}" ]];then
  echo -n 'Username: '
  read -r username
  echo -n 'Password: '
  read -r password

  echo -n "Downloading ShowHowTo test set image sequences ... "
  wget --user="${username}" --password="${password}" https://data.ciirc.cvut.cz/public/projects/2024ShowHowTo/dataset_sequences/ShowHowToTest.images.tar --no-check-certificate -q
  tar xf ShowHowToTest.images.tar && rm ShowHowToTest.images.tar && echo "OK ✓" || echo "ERROR ✗"

  echo -n "Downloading ShowHowTo train set image sequences ... "
  wget --user="${username}" --password="${password}" https://data.ciirc.cvut.cz/public/projects/2024ShowHowTo/dataset_sequences/ShowHowToTrain.images.tar --no-check-certificate -q
  tar xf ShowHowToTrain.images.tar && rm ShowHowToTrain.images.tar && echo "OK ✓" || echo "ERROR ✗"
fi
