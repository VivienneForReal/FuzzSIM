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

## Data installation
If you want to keep the data locally, please run the following command:
```bash
chmod +x scripts/data_installer.sh
./scripts/data_installer.sh