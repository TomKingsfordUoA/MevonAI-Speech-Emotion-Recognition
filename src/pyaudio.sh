sudo apt-get purge --remove python3-pyaudio
cd portaudio/
./configure
make
make install
