# ColorNetwork
Maybe a TN for lattice gauge theory

# Dependency

Cuda

cuTensor

to install cuTensor:
https://developer.nvidia.com/cutensor/downloads

Ubuntu 18.04 x86_64
	sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
	sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub	
	sudo apt update	
	sudo apt -y install libcutensor1 libcutensor-dev libcutensor-doc	
Ubuntu 18.04 ppc64le
	sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/ppc64el/ /"	
	sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/ppc64el/7fa2af80.pub	
	sudo apt update	
	sudo apt -y install libcutensor1 libcutensor-dev libcutensor-doc	
Ubuntu 18.04 sbsa
	sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/sbsa/ /"	
	sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/sbsa/7fa2af80.pub	
	sudo apt update	
	sudo apt -y install libcutensor1 libcutensor-dev libcutensor-doc
Ubuntu 20.04
	sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
	sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
	sudo apt update	
	sudo apt -y install libcutensor1 libcutensor-dev libcutensor-doc

cuTensor require a sufficiently recent (GCC 5 or higher) libstdc++ when linking statically.

