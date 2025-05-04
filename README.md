# FuzzSIM

## Project structure
```txt
FuzzSIM/
│── src/            
│   ├── classif/
│   │   ├── base.py
│   │   ├── knn.py      
│   │   ├── eval.py  
│   ├── fuzz/     
│   │   ├── capacity.py              
│   │   ├── choquet.py              
│   │   ├── norm.py              
│   │   ├── sim.py              
│   ├── utils/     
│   │   ├── dataloader.py              
│   │   ├── set.py              
│   │   ├── utils.py              
│   │   ├── visualization.py    
│   ├── fsim.py
│── configs/               
│   ├── config.yml    
│── bin/                 
│── scripts/                
│   ├── launch.sh     
│── environment.yml
│── README.md               
│── .gitignore  
│── .gitattributes             
```


## Setup conda environment
### Quick setup
```bash
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

## Manual setup
```bash
# Create a conda environment with compatible python version (3.10)
conda create -n fuzzsim python=3.10 -y

# Activate the environment
conda activate fuzzsim

# Optional: upgrade pip
pip install --upgrade pip

# Install other dependencies
pip install -r requirements.txt
```