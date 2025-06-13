# FuzzSIM

## Project structure
```txt
FuzzSIM/
│── fuzz/            
│   ├── choquet/
│   │   ├── __init__.py
│   │   ├── choquet.py
│   │   ├── classic.py      
│   │   ├── d_choquet.py  
│   │   ├── linear_d_choquet.py  
│   ├── src/     
│   │   ├── __init__.py
│   │   ├── base.py 
│   │   ├── knn.py              
│   │   ├── norm.py              
│   │   ├── sim.py              
│   ├── __init__.py
│   ├── set.py              
│   ├── dataloader.py              
│   ├── eval.py    
│   ├── optim.py
│   ├── utils.py 
│── configs/                
│   ├── configs.yml  
│── scripts/                
│   ├── setup_env.sh     
│── __init__.py
│── requirements.txt
│── LICENSE
│── README.md                      
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